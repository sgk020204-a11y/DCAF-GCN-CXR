# -*- coding: utf-8 -*-

import argparse
import contextlib
import json
import os
import pickle
import random
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from coco import COCO2014
from engine2 import GCNMultiLabelMAPEngine as _BaseEngine
from losses import CombinedMultiLabelLoss
from models2 import gcn_SQNET_ViT


def _strip_prefix_if_present(state_dict, prefix: str):
    if not prefix:
        return state_dict
    stripped = {}
    for k, v in state_dict.items():
        stripped[k[len(prefix):] if k.startswith(prefix) else k] = v
    return stripped


def load_checkpoint_to_core(core_model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[resume] checkpoint not found: {ckpt_path}")

    print(f"=> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
    sd = _strip_prefix_if_present(sd, "module.")
    sd = _strip_prefix_if_present(sd, "core.")
    missing, unexpected = core_model.load_state_dict(sd, strict=False)
    print(f"=> Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("   missing keys (show up to 10):", missing[:10])
    if len(unexpected) > 0:
        print("   unexpected keys (show up to 10):", unexpected[:10])


def set_random_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


class ModelInputAdapter(nn.Module):
    """
    Ensure the model always treats the last positional argument as inp.
    Compatible with:
      - model(images, inp)
      - model(images, name, inp)
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, images, *args, **kwargs):
        if len(args) == 0:
            raise TypeError("ModelInputAdapter expects at least one extra positional arg as inp")
        inp = args[-1]
        return self.core(images, inp, **kwargs)

    def __getattr__(self, name):
        if name in ("core",):
            return super().__getattr__(name)
        return getattr(self.core, name)


class AuxWithVisSemLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, model: nn.Module, w_vis: float = 0.2, w_sem: float = 0.2):
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.w_vis = float(w_vis)
        self.w_sem = float(w_sem)

    def forward(self, logits_main: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(logits_main, targets)
        logits_vis = getattr(self.model, "_last_logits_vis", None)
        if (logits_vis is not None) and (self.w_vis != 0.0):
            loss = loss + self.w_vis * self.base_loss(logits_vis, targets)
        logits_sem = getattr(self.model, "_last_logits_sem", None)
        if (logits_sem is not None) and (self.w_sem != 0.0):
            loss = loss + self.w_sem * self.base_loss(logits_sem, targets)
        return loss


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        assert 0.0 < float(decay) < 1.0
        self.decay = float(decay)
        self.shadow = {}
        self._init_from(model)

    @torch.no_grad()
    def _init_from(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if torch.is_tensor(v) and v.is_floating_point():
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        sd = model.state_dict()
        for k, ema_v in self.shadow.items():
            v = sd.get(k, None)
            if v is None or (not torch.is_tensor(v)) or (not v.is_floating_point()):
                continue
            ema_v.mul_(d).add_(v.detach(), alpha=(1.0 - d))

    @contextlib.contextmanager
    def swap_to_ema(self, model: nn.Module, backup_on_cpu: bool = False):
        sd = model.state_dict()
        backup = {}
        try:
            for k, ema_v in self.shadow.items():
                if k not in sd:
                    continue
                v = sd[k]
                if (not torch.is_tensor(v)) or (not v.is_floating_point()):
                    continue
                backup[k] = v.detach().cpu().clone() if backup_on_cpu else v.detach().clone()
                v.copy_(ema_v.to(device=v.device, dtype=v.dtype))
            yield
        finally:
            for k, old in backup.items():
                v = sd[k]
                if old.device != v.device:
                    old = old.to(v.device)
                v.copy_(old)


class EMAGCNMultiLabelMAPEngine(_BaseEngine):
    def __init__(self, state, ema_decay: float = 0.9999, ema_eval: bool = True, ema_backup_on_cpu: bool = False):
        super().__init__(state)
        self.ema_decay = float(ema_decay)
        self.ema_eval = bool(ema_eval)
        self.ema_backup_on_cpu = bool(ema_backup_on_cpu)
        self.ema = None
        self._ema_model_ref = None

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()

        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        output = model(feature_var, inp_var)
        raw_loss = criterion(output, target_var)

        self.state['output'] = output
        self.state['loss'] = raw_loss

        if training:
            if self.ema is None:
                self.ema = EMA(model, decay=self.ema_decay)
            if self._ema_model_ref is None:
                self._ema_model_ref = model

            accum_steps = max(1, int(self.state.get('accum_steps', 1)))
            cur_iter = int(self.state.get('iteration', 0))
            total_iters = int(self.state.get('num_steps_epoch', 0))

            if (cur_iter % accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)

            (raw_loss / float(accum_steps)).backward()

            should_step = ((cur_iter + 1) % accum_steps == 0) or ((cur_iter + 1) == total_iters)
            if should_step:
                grad_clip = float(self.state.get('grad_clip', 10.0))
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                self.ema.update(model)

            self.state['loss'] = raw_loss

    def validate(self, data_loader, model, criterion):
        if self._ema_model_ref is None:
            self._ema_model_ref = model
        if self.ema_eval and (self.ema is not None):
            with self.ema.swap_to_ema(model, backup_on_cpu=self.ema_backup_on_cpu):
                return super().validate(data_loader, model, criterion)
        return super().validate(data_loader, model, criterion)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_in_channel(train_dataset, fallback_word_emb_path: str) -> int:
    if hasattr(train_dataset, "inp"):
        try:
            arr = np.asarray(train_dataset.inp)
            if arr.ndim == 2:
                return int(arr.shape[1])
        except Exception:
            pass

    if fallback_word_emb_path and os.path.isfile(fallback_word_emb_path):
        try:
            with open(fallback_word_emb_path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, dict):
                for k in ("emb", "embedding", "embeddings", "word_embedding", "vectors", "W"):
                    if k in raw:
                        mat = np.asarray(raw[k])
                        if mat.ndim == 2:
                            return int(mat.shape[1])
            else:
                mat = np.asarray(raw)
                if mat.ndim == 2:
                    return int(mat.shape[1])
        except Exception:
            pass
    return 300


def _load_train_label_stats(data_path: str, num_classes: int):
    anno_path = Path(data_path) / "train_anno_aligned.json"
    if not anno_path.exists():
        return None, None
    try:
        records = _load_json(anno_path)
        counts = np.zeros(num_classes, dtype=np.float64)
        for item in records:
            labels = item.get("labels", [])
            if labels is None:
                continue
            for c in labels:
                ci = int(c)
                if 0 <= ci < num_classes:
                    counts[ci] += 1.0
        return counts, float(len(records))
    except Exception as e:
        print(f"[WARN] failed to parse training annotations for class stats: {e}")
        return None, None


def _build_class_weights(args, num_classes: int, device: torch.device):
    label_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    pos_weight = None

    if not args.auto_class_weights:
        return label_weights, pos_weight

    counts, total = _load_train_label_stats(args.data_path, num_classes)
    if counts is None or total is None or total <= 0:
        print("[WARN] auto class weighting fallback to uniform (stats unavailable).")
        return label_weights, pos_weight

    pos = np.clip(counts, 0.0, total)
    neg = np.maximum(total - pos, 0.0)

    # BCE pos_weight
    pos_w = ((neg + 1.0) / (pos + 1.0)) ** float(args.pos_weight_power)
    pos_w = np.clip(pos_w, float(args.pos_weight_min), float(args.pos_weight_max))
    pos_weight = torch.tensor(pos_w, dtype=torch.float32, device=device)

    # label_weights for global reweighting
    inv = ((total + num_classes) / (pos + 1.0)) ** float(args.label_weight_power)
    inv = inv / (np.mean(inv) + 1e-12)
    inv = np.clip(inv, float(args.label_weight_min), float(args.label_weight_max))
    label_weights = torch.tensor(inv, dtype=torch.float32, device=device)

    print("[AutoWeight] class stats loaded from train_anno_aligned.json")
    print("[AutoWeight] pos_count min/max = {:.0f}/{:.0f}".format(float(np.min(pos)), float(np.max(pos))))
    print("[AutoWeight] pos_weight min/max = {:.3f}/{:.3f}".format(float(np.min(pos_w)), float(np.max(pos_w))))
    print("[AutoWeight] label_weight min/max = {:.3f}/{:.3f}".format(float(np.min(inv)), float(np.max(inv))))
    return label_weights, pos_weight


def _parse_args():
    parser = argparse.ArgumentParser(description="ML-GCN + ViT-Hybrid ablation trainer")

    # Paths
    parser.add_argument("--data-path", type=str, default="./data/chestxray14")
    parser.add_argument("--adj-file", type=str, default="./graph/Xray_adj.pkl")
    parser.add_argument("--word-emb", type=str, default="./graph/word14.pkl")
    parser.add_argument("--label-emb-path", type=str, default="", help="default uses --word-emb")
    parser.add_argument("--mae-ckpt", type=str, default="")
    parser.add_argument("--densenet-ckpt", type=str, default="")
    parser.add_argument("--save-model-path", type=str, default="./outputs/checkpoints/chestxray/")

    # Training core
    parser.add_argument("--image-size", default=448, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--accum-steps", default=1, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--epoch_step", default=[6, 12, 18], type=int, nargs="+")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device-ids", default=[0], type=int, nargs="+")
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dropout-prob", default=0.3, type=float)

    # LR / optimizer
    parser.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", default=0.006, type=float, help="reference lr at base-batch-size")
    parser.add_argument("--base-batch-size", default=64, type=int, help="for lr auto scaling")
    parser.add_argument("--auto-scale-lr", dest="auto_scale_lr", action="store_true", default=True)
    parser.add_argument("--no-auto-scale-lr", dest="auto_scale_lr", action="store_false")
    parser.add_argument("--lrp", default=0.2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="backbone wd")
    parser.add_argument("--wd-task", default=1e-4, type=float, help="task module wd")
    parser.add_argument("--wd-head", default=1e-4, type=float, help="classification head wd")
    parser.add_argument("--lr-scheduler", default="step", choices=["step", "cosine"])
    parser.add_argument("--lr-decay", default=0.2, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--min-lr-ratio", default=0.05, type=float)
    parser.add_argument("--grad-clip", default=10.0, type=float)
    parser.add_argument("--mona-lr-mult", default=1.0, type=float)
    parser.add_argument("--dense-lr-mult", default=2.0, type=float)
    parser.add_argument("--align-lr-mult", default=1.0, type=float)
    parser.add_argument("--head-lr-mult", default=1.0, type=float)
    parser.add_argument("--ms-fuse-lr-mult", default=1.0, type=float)

    # Augmentation
    parser.add_argument("--aug-resized-min", default=0.85, type=float)
    parser.add_argument("--aug-rotation", default=15.0, type=float)
    parser.add_argument("--aug-jitter", default=0.2, type=float)
    parser.add_argument("--aug-erase-prob", default=0.12, type=float)
    parser.add_argument("--aug-erase-min", default=0.02, type=float)
    parser.add_argument("--aug-erase-max", default=0.12, type=float)

    # Loss
    parser.add_argument("--w-asl", default=1.0, type=float)
    parser.add_argument("--w-bce", default=0.1, type=float)
    parser.add_argument("--w-gsl", default=0.03, type=float)
    parser.add_argument("--w-auc", default=0.06, type=float)
    parser.add_argument("--w-vis", default=0.2, type=float)
    parser.add_argument("--w-sem", default=0.2, type=float)
    parser.add_argument("--auc-warmup-epochs", default=5, type=int)
    parser.add_argument("--auc-warmup-start", default=0.0, type=float)
    parser.add_argument("--gamma-neg", default=4.0, type=float)
    parser.add_argument("--gamma-pos", default=1.0, type=float)
    parser.add_argument("--clip", default=0.05, type=float)
    parser.add_argument("--eps", default=0.05, type=float)
    parser.add_argument("--auto-class-weights", dest="auto_class_weights", action="store_true", default=True)
    parser.add_argument("--no-auto-class-weights", dest="auto_class_weights", action="store_false")
    parser.add_argument("--pos-weight-power", default=1.0, type=float)
    parser.add_argument("--pos-weight-min", default=1.0, type=float)
    parser.add_argument("--pos-weight-max", default=10.0, type=float)
    parser.add_argument("--label-weight-power", default=0.5, type=float)
    parser.add_argument("--label-weight-min", default=0.5, type=float)
    parser.add_argument("--label-weight-max", default=2.5, type=float)

    # EMA
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--no-ema-eval", action="store_true")
    parser.add_argument("--ema-backup-on-cpu", action="store_true")

    # Progress / logging
    parser.add_argument("--use-pb", dest="use_pb", action="store_true", default=True)
    parser.add_argument("--no-use-pb", dest="use_pb", action="store_false")
    parser.add_argument("--print-freq", default=0, type=int)
    parser.add_argument("--pb-postfix-freq", default=20, type=int)
    parser.add_argument("--pb-mininterval", default=1.0, type=float)
    parser.add_argument("--pb-miniters", default=10, type=int)
    parser.add_argument("--disable-checkpoint", action="store_true")
    parser.add_argument("--save-last-too", action="store_true", help="save every epoch checkpoint in addition to best")

    # Ablation switches
    parser.add_argument("--use-dense-overlock", action="store_true", help="enable OverLoCK/ContMix")
    parser.add_argument("--fusion-type", default="dcaf_bi", choices=["dcaf_bi", "dcaf_v2c", "dcaf_c2v", "concat", "weighted_sum"])
    parser.add_argument("--fusion-scales", default="c3,c4", help="comma separated from c1,c2,c3,c4")
    parser.add_argument("--graph-source", default="coocc_only", choices=["coocc_only", "semantic_only", "joint", "identity"])
    parser.add_argument("--graph-alpha", default=0.5, type=float)
    parser.add_argument("--graph-edge-policy", default="threshold", choices=["threshold", "topk"])
    parser.add_argument("--graph-topk", default=0, type=int)

    return parser.parse_args()


def main_Chestxray():
    args = _parse_args()
    set_random_seed(seed=int(args.seed), deterministic=bool(args.deterministic))

    fusion_scales = [s.strip().lower() for s in str(args.fusion_scales).split(",") if s.strip()]
    if len(fusion_scales) == 0:
        raise ValueError("--fusion-scales cannot be empty")
    valid_scales = {"c1", "c2", "c3", "c4"}
    bad_scales = [s for s in fusion_scales if s not in valid_scales]
    if bad_scales:
        raise ValueError(f"Unsupported fusion scales: {bad_scales}, valid={sorted(valid_scales)}")
    if args.graph_edge_policy == "topk" and args.graph_topk <= 0:
        raise ValueError("--graph-topk must be > 0 when --graph-edge-policy=topk")
    if len(args.device_ids) == 0:
        raise ValueError("--device-ids cannot be empty")

    use_gpu = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device_ids[0]}" if use_gpu else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomResizedCrop(args.image_size, scale=(args.aug_resized_min, 1.0)),
        transforms.RandomRotation(degrees=float(args.aug_rotation)),
        transforms.ColorJitter(brightness=float(args.aug_jitter), contrast=float(args.aug_jitter)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(
            p=float(args.aug_erase_prob),
            scale=(float(args.aug_erase_min), float(args.aug_erase_max)),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = COCO2014(root=args.data_path, phase="train", inp_name=args.word_emb, transform=train_transform)
    val_dataset = COCO2014(root=args.data_path, phase="val", inp_name=args.word_emb, transform=val_transform)

    with open(args.adj_file, "rb") as f:
        adj = pickle.load(f)
    label_names = list(adj["classes"])
    num_classes = int(len(label_names))

    cat_path = Path(args.data_path) / "category_aligned.json"
    if cat_path.exists():
        cat = _load_json(cat_path)
        classes_adj = [c.lower() for c in label_names]
        classes_cat = [k for k, _ in sorted(cat.items(), key=lambda x: x[1])]
        if classes_adj != classes_cat:
            raise RuntimeError(f"Label order mismatch between adj and category file.\nadj={classes_adj}\ncat={classes_cat}")

    label_emb_path = args.label_emb_path if args.label_emb_path else args.word_emb
    in_channel = _infer_in_channel(train_dataset, fallback_word_emb_path=args.word_emb)

    model_core = gcn_SQNET_ViT(
        num_classes=num_classes,
        t=0.4,
        adj_file=args.adj_file,
        in_channel=in_channel,
        mae_ckpt_path=args.mae_ckpt,
        densenet_checkpoint_path=args.densenet_ckpt,
        model_name="vit_base_patch16_384",
        use_dense_overlock=bool(args.use_dense_overlock),
        fusion_type=args.fusion_type,
        fusion_scales=tuple(fusion_scales),
        graph_source=args.graph_source,
        graph_alpha=args.graph_alpha,
        graph_edge_policy=args.graph_edge_policy,
        graph_topk=args.graph_topk,
        label_emb_path=label_emb_path,
    )

    if args.resume:
        load_checkpoint_to_core(model_core, args.resume)

    model = ModelInputAdapter(model_core).to(device)

    # Build adaptive class weights from training annotations
    label_weights, pos_weight = _build_class_weights(args, num_classes=num_classes, device=device)
    A_for_graph = getattr(model, "A", None)
    if isinstance(A_for_graph, torch.Tensor):
        A_for_graph = A_for_graph.detach().to(device)

    base_criterion = CombinedMultiLabelLoss(
        w_asl=float(args.w_asl),
        w_bce=float(args.w_bce),
        w_gsl=float(args.w_gsl),
        w_auc=float(args.w_auc),
        gamma_pos=float(args.gamma_pos),
        gamma_neg=float(args.gamma_neg),
        clip=float(args.clip),
        eps=float(args.eps),
        pos_weight=pos_weight,
        label_weights=label_weights,
        A=A_for_graph,
        A_normalized=True,
        auc_max_pos=32,
        auc_max_neg=64,
        auc_margin=0.0,
    ).to(device)

    criterion = AuxWithVisSemLoss(
        base_loss=base_criterion,
        model=model,
        w_vis=float(args.w_vis),
        w_sem=float(args.w_sem),
    ).to(device)

    effective_batch = int(args.batch_size) * max(1, int(args.accum_steps))
    if args.auto_scale_lr:
        run_lr = float(args.lr) * float(effective_batch) / float(max(1, int(args.base_batch_size)))
    else:
        run_lr = float(args.lr)

    print(
        "[Config] use_gpu={}, device={}, device_ids={}, seed={}, deterministic={}".format(
            use_gpu, device, args.device_ids, args.seed, bool(args.deterministic)
        )
    )
    print(
        "[Config] batch_size={}, accum_steps={}, effective_batch={}, base_batch_size={}, auto_scale_lr={}, ref_lr={}, run_lr={:.6f}".format(
            args.batch_size, args.accum_steps, effective_batch, args.base_batch_size,
            bool(args.auto_scale_lr), args.lr, run_lr
        )
    )
    print(
        "[Ablation] fusion_type={}, fusion_scales={}, graph_source={}, graph_edge_policy={}, graph_topk={}".format(
            args.fusion_type, fusion_scales, args.graph_source, args.graph_edge_policy, args.graph_topk
        )
    )

    param_groups = model.get_config_optim(
        lr=run_lr,
        lrp=args.lrp,
        wd_backbone=args.weight_decay,
        wd_task=args.wd_task,
        wd_head=args.wd_head,
        mona_lr_mult=args.mona_lr_mult,
        dense_lr_mult=args.dense_lr_mult,
        align_lr_mult=args.align_lr_mult,
        head_lr_mult=args.head_lr_mult,
        ms_fuse_lr_mult=args.ms_fuse_lr_mult,
    )

    if args.opt.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=run_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,  # use per-group wd
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=run_lr,
            momentum=float(args.momentum),
            nesterov=True,
            weight_decay=0.0,  # use per-group wd
        )

    state = {
        "use_gpu": use_gpu,
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "max_epochs": int(args.epochs),
        "evaluate": bool(args.evaluate),
        "resume": "",  # already loaded above
        "num_classes": num_classes,
        "save_model_path": args.save_model_path,
        "workers": int(args.workers),
        "epoch_step": list(args.epoch_step),
        "lr": float(run_lr),
        "device_ids": list(args.device_ids),
        "use_pb": bool(args.use_pb),
        "print_freq": int(args.print_freq),
        "pb_postfix_freq": int(args.pb_postfix_freq),
        "pb_mininterval": float(args.pb_mininterval),
        "pb_miniters": int(args.pb_miniters),
        "accum_steps": max(1, int(args.accum_steps)),
        "grad_clip": float(args.grad_clip),
        "lr_scheduler": str(args.lr_scheduler),
        "lr_decay": float(args.lr_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
        "train_transform": train_transform,
        "val_transform": val_transform,
        "label_names": label_names,
        "disable_checkpoint": bool(args.disable_checkpoint),
        "save_best_only": (not bool(args.save_last_too)),
    }

    if args.use_ema:
        engine = EMAGCNMultiLabelMAPEngine(
            state,
            ema_decay=float(args.ema_decay),
            ema_eval=(not bool(args.no_ema_eval)),
            ema_backup_on_cpu=bool(args.ema_backup_on_cpu),
        )
    else:
        engine = _BaseEngine(state)

    # Warm up the AUC component to avoid early-stage instability.
    _orig_on_start_epoch = getattr(engine, "on_start_epoch", None)
    if _orig_on_start_epoch is not None and hasattr(base_criterion, "w_auc"):
        target_w_auc = float(args.w_auc)
        warmup_epochs = max(0, int(args.auc_warmup_epochs))
        start_w_auc = float(args.auc_warmup_start)

        def _on_start_epoch_patched(self, training, model_, criterion_, data_loader_, optimizer_=None, display=True):
            ret = _orig_on_start_epoch(training, model_, criterion_, data_loader_, optimizer_, display)
            if training:
                ep = int(self.state.get("epoch", 0))
                if warmup_epochs <= 0:
                    base_criterion.w_auc = target_w_auc
                elif ep >= warmup_epochs:
                    base_criterion.w_auc = target_w_auc
                else:
                    alpha = float(ep + 1) / float(warmup_epochs)
                    base_criterion.w_auc = start_w_auc + alpha * (target_w_auc - start_w_auc)
            return ret

        engine.on_start_epoch = types.MethodType(_on_start_epoch_patched, engine)

    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

    if args.use_ema and hasattr(engine, "ema") and (engine.ema is not None):
        os.makedirs(args.save_model_path, exist_ok=True)
        ema_path = os.path.join(args.save_model_path, "ema_last.pth")
        model_ref = getattr(engine, "_ema_model_ref", None) or model
        with engine.ema.swap_to_ema(model_ref, backup_on_cpu=bool(args.ema_backup_on_cpu)):
            sd = model_ref.module.state_dict() if hasattr(model_ref, "module") else model_ref.state_dict()
            torch.save({"state_dict": sd, "ema_decay": float(args.ema_decay)}, ema_path)
        print(f"[EMA] saved to: {ema_path}")


if __name__ == "__main__":
    main_Chestxray()
