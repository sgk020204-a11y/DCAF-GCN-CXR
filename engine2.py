import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from util import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm


tqdm.monitor_interval = 0

from util import MultiScaleCrop, Warp


class Engine(object):
    def __init__(self, state={}):
        # -------- 默认参数（精简 setdefault） --------
        self.state = state
        s = self.state
        s.setdefault('use_gpu', torch.cuda.is_available())
        s.setdefault('image_size', 448)
        s.setdefault('batch_size', 64)
        s.setdefault('workers', 25)
        s.setdefault('device_ids', None)
        s.setdefault('evaluate', False)
        s.setdefault('start_epoch', 0)
        s.setdefault('max_epochs', 90)
        s.setdefault('epoch_step', [])
        s.setdefault('print_freq', 0)
        s.setdefault('use_pb', True)
        s.setdefault('pb_postfix_freq', 20)
        s.setdefault('pb_mininterval', 1.0)
        s.setdefault('pb_miniters', 10)
        s.setdefault('accum_steps', 1)
        s.setdefault('grad_clip', 10.0)
        s.setdefault('lr_scheduler', 'step')
        s.setdefault('lr_decay', 0.2)
        s.setdefault('warmup_epochs', 0)
        s.setdefault('min_lr_ratio', 0.05)
        s.setdefault('save_best_only', True)
        s.setdefault('disable_checkpoint', False)

        # 注意：best_score 语义为 “验证集 Macro-AUC 最优值”
        s.setdefault('best_score', 0.0)

        # meters & time
        s['meter_loss'] = tnt.meter.AverageValueMeter()
        s['batch_time'] = tnt.meter.AverageValueMeter()
        s['data_time'] = tnt.meter.AverageValueMeter()

    def _state(self, name):
        return self.state.get(name, None)

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\tLoss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test:\tLoss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # record loss
        self.state['loss_batch'] = float(self.state['loss'].item())
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print(
                    'Epoch: [{0}][{1}/{2}]\tTime {bt_c:.3f} ({bt:.3f})\tData {dt_c:.3f} ({dt:.3f})\tLoss {lc:.4f} ({l:.4f})'
                    .format(self.state['epoch'], self.state['iteration'], len(data_loader),
                            bt_c=self.state['batch_time_current'], bt=batch_time,
                            dt_c=self.state['data_time_batch'], dt=data_time,
                            lc=self.state['loss_batch'], l=loss))
            else:
                print(
                    'Test: [{0}/{1}]\tTime {bt_c:.3f} ({bt:.3f})\tData {dt_c:.3f} ({dt:.3f})\tLoss {lc:.4f} ({l:.4f})'
                    .format(self.state['iteration'], len(data_loader),
                            bt_c=self.state['batch_time_current'], bt=batch_time,
                            dt_c=self.state['data_time_batch'], dt=data_time,
                            lc=self.state['loss_batch'], l=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None):
        """
        保持与你现有脚本的兼容：兼容 (images, extra) / images 两种输入形式；
        将本 batch 的 output 与 loss 写回 self.state 供后续使用。
        """
        for i, (inp, target) in enumerate(data_loader):
            self.state['input'] = inp
            self.state['target'] = target

            # 适配输入到设备
            if isinstance(inp, (tuple, list)):
                inputs = tuple(torch.autograd.Variable(x.cuda(non_blocking=True) if torch.cuda.is_available() else x)
                               for x in inp)
                self.state['input'] = inputs
            else:
                self.state['input'] = torch.autograd.Variable(
                    inp.cuda(non_blocking=True) if torch.cuda.is_available() else inp
                )

            self.state['target'] = torch.autograd.Variable(
                target.cuda(non_blocking=True) if torch.cuda.is_available() else target
            )

            # 前向 & 反向
            if training:
                optimizer.zero_grad(set_to_none=True)
                output = model(*self.state['input']) if isinstance(self.state['input'], tuple) else model(self.state['input'])
                loss = criterion(output, self.state['target'])
                loss.backward()
                # 可选：梯度裁剪（若需要）
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            else:
                with torch.no_grad():
                    output = model(*self.state['input']) if isinstance(self.state['input'], tuple) else model(self.state['input'])
                    loss = criterion(output, self.state['target'])

            # 缓存
            self.state['output'] = output
            self.state['loss'] = loss

    def init_learning(self, model, criterion):
        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=getattr(model, 'image_normalization_mean', [0.485, 0.456, 0.406]),
                                             std=getattr(model, 'image_normalization_std', [0.229, 0.224, 0.225]))
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=getattr(model, 'image_normalization_mean', [0.485, 0.456, 0.406]),
                                             std=getattr(model, 'image_normalization_std', [0.229, 0.224, 0.225]))
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
        """
        完整训练-验证-保存流程：
        1) 构建 dataloader 与数据增强
        2) 自动从数据集提取 label_names（若存在），用于逐类 AUROC 的中文/英文名打印
        3) （可选）恢复检查点
        4) GPU 并行封装
        5) 训练/验证循环；以 Macro-AUC 作为 “best” 的依据并保存
        """
        # 1) 数据增强 & DataLoader
        self.init_learning(model, criterion)

        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.state['batch_size'],
            shuffle=True,
            num_workers=self.state['workers'],
            pin_memory=self.state['use_gpu']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.state['batch_size'],
            shuffle=False,
            num_workers=self.state['workers'],
            pin_memory=self.state['use_gpu']
        )

        # 2) 自动提取并打印标签名（供 validate() 的 Per-label AUROC 使用）
        if self.state.get('label_names') is None:
            def _extract_names(ds):
                # 优先顺序覆盖常见字段
                for attr in ['classes', 'class_names', 'label_names', 'idx_to_class', 'idx2label', 'labels']:
                    if hasattr(ds, attr):
                        obj = getattr(ds, attr)
                        # dict: 可能是 {idx: name} 或 {name: idx}
                        if isinstance(obj, dict):
                            if all(isinstance(k, int) for k in obj.keys()):
                                # {idx: name}
                                return [obj[i] for i in range(len(obj))]
                            elif all(isinstance(v, int) for v in obj.values()):
                                # {name: idx}
                                inv = {v: k for k, v in obj.items()}
                                return [inv[i] for i in range(len(inv))]
                        elif isinstance(obj, (list, tuple)):
                            return list(obj)
                # 常见的 label_map: {name: idx}
                if hasattr(ds, 'label_map') and isinstance(ds.label_map, dict):
                    inv = {v: k for k, v in ds.label_map.items()}
                    return [inv[i] for i in range(len(inv))]
                return None

            names = _extract_names(train_dataset) or _extract_names(val_dataset)
            if names is not None:
                self.state['label_names'] = list(names)
                print('=== Index -> Label used by the dataloader ===')
                for i, n in enumerate(self.state['label_names']):
                    print(f'{i:02d}: {n}')
            else:
                print('警告：未在数据集中发现标签名属性，将使用 Label_0... 的占位名。')

        # 3) （可选）恢复检查点
        if self._state('resume') is not None and self.state['resume']:
            if os.path.isfile(self.state['resume']):
                print("=> 加载检查点 '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint.get('epoch', 0)
                self.state['best_score'] = checkpoint.get('best_score', 0.0)
                model.load_state_dict(checkpoint['state_dict'])
                print("=> 成功加载检查点 (epoch {})".format(self.state['start_epoch']))
            else:
                print("=> 未找到检查点 '{}'".format(self.state['resume']))

        # 4) GPU 并行
        if self.state['use_gpu']:
            cudnn.benchmark = True
            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            criterion = criterion.cuda()

        # 仅评估
        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return self.state['best_score']

        # 5) 训练-验证循环（以 Macro-AUC 选最优）
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('当前学习率:', lr)
            print('当前epoch:', epoch)

            # 训练一个 epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # 验证：返回 Macro-AUC（作为保存 best 的依据）
            auc_macro = self.validate(val_loader, model, criterion)

            # 保存最佳（基于 Macro-AUC）
            is_best = auc_macro > self.state['best_score']
            if is_best:
                self.state['best_score'] = auc_macro

            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],  # 注意：这里保存的是 AUC
            }, is_best)

            print(' *** 当前最佳(AUC) = {best:.4f}'.format(best=self.state['best_score']))

        return self.state['best_score']


    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        self.state['num_steps_epoch'] = len(data_loader)
        use_pb = bool(self.state.get('use_pb', True))
        pb_postfix_freq = max(1, int(self.state.get('pb_postfix_freq', 20)))
        if use_pb:
            max_epochs = int(self.state.get('max_epochs', 0))
            data_loader = tqdm(
                data_loader,
                desc=f'Train {epoch + 1}/{max_epochs}',
                dynamic_ncols=True,
                leave=True,
                mininterval=float(self.state.get('pb_mininterval', 1.0)),
                miniters=int(self.state.get('pb_miniters', 10)),
                position=0,
            )

        end = time.time()
        for i, (inp, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = inp
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(True, model, criterion, data_loader, optimizer)

            if use_pb and (((i + 1) % pb_postfix_freq == 0) or ((i + 1) == len(data_loader))):
                avg_loss = self.state['meter_loss'].value()[0]
                cur_loss = float(self.state.get('loss_batch', 0.0))
                lr = float(optimizer.param_groups[0].get('lr', 0.0))
                pct = 100.0 * float(i + 1) / max(1.0, float(len(data_loader)))
                data_loader.set_postfix_str(
                    f'{pct:5.1f}% | loss {cur_loss:.4f} | avg {avg_loss:.4f} | lr {lr:.2e}'
                )

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        """
        验证阶段：计算 Macro-AUC / mAP，打印逐类 AUROC。
        返回：auc_macro（用于保存最佳）
        额外功能（可选）：
        - self.state['cam_vis']=True 时，在验证集第一个 batch 上导出“每个标签一张热力图”的网格图；
        - 输出目录：self.state.get('cam_save_dir', './cam_out')
        """
        import os
        import numpy as np
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from sklearn.metrics import roc_auc_score, average_precision_score

        # --------- 一些小工具：递归搬运到 GPU / 解包 DataParallel ----------
        def _unwrap(m):
            return m.module if hasattr(m, "module") else m

        def _to_device(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device, non_blocking=True)
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_device(x, device) for x in obj)
            return obj  # 非 tensor 原样返回（如文件名等）

        def _call_model(core_model, batch_inp):
            # batch_inp 可能是 (images, inp) 这样的 tuple/list，也可能是单 tensor
            if isinstance(batch_inp, (list, tuple)):
                return core_model(*batch_inp)
            return core_model(batch_inp)

        def _extract_images(batch_inp):
            # 默认第 0 个就是 images（你工程里就是这样喂给 model(images, inp)）
            if isinstance(batch_inp, (list, tuple)) and len(batch_inp) > 0:
                return batch_inp[0]
            return batch_inp

        # --------- CAM 可视化（不依赖 cv2） ----------
        class _GradCAM2D:
            """针对输出为 (B,C,h,w) 的层做标准 Grad-CAM。"""
            def __init__(self, core_model, target_module):
                self.core_model = core_model
                self.target_module = target_module
                self.activations = None
                self.gradients = None
                self._register()

            def _register(self):
                def fwd_hook(m, inp, out):
                    self.activations = out

                def bwd_hook(m, grad_in, grad_out):
                    self.gradients = grad_out[0]

                self.target_module.register_forward_hook(fwd_hook)
                self.target_module.register_full_backward_hook(bwd_hook)

            def cam_for_class(self, batch_inp, class_idx: int, sample_idx: int = 0):
                self.core_model.zero_grad(set_to_none=True)
                logits = _call_model(self.core_model, batch_inp)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]  # 兼容 return_all=True 的情况

                score = logits[sample_idx, class_idx]
                score.backward(retain_graph=True)

                acts = self.activations[sample_idx]  # (C,h,w)
                grads = self.gradients[sample_idx]   # (C,h,w)
                weights = grads.mean(dim=(1, 2), keepdim=True)  # (C,1,1)
                cam_map = (weights * acts).sum(dim=0)           # (h,w)
                cam_map = torch.relu(cam_map)

                cam_map = cam_map.detach().cpu().numpy()
                cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-6)
                return cam_map

        def _denorm_to_uint8(images, mean, std):
            """
            images: (B,3,H,W) normalized
            return: (B,H,W,3) uint8
            """
            mean_t = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
            std_t = torch.tensor(std, device=images.device).view(1, 3, 1, 1)
            x = images * std_t + mean_t
            x = torch.clamp(x, 0, 1)
            x = (x * 255.0).byte()
            x = x.permute(0, 2, 3, 1).contiguous()
            return x

        def _overlay_cam(rgb_uint8, cam_small, alpha=0.45):
            """
            rgb_uint8: (H,W,3) uint8
            cam_small: (h,w) float [0,1]
            """
            H, W = rgb_uint8.shape[:2]
            cam_t = torch.from_numpy(cam_small)[None, None, ...].float()  # (1,1,h,w)
            cam_rs = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
            cam_rs = np.clip(cam_rs, 0, 1)

            heat = (cm.get_cmap("jet")(cam_rs)[..., :3] * 255.0).astype(np.uint8)  # (H,W,3)
            out = (alpha * heat + (1 - alpha) * rgb_uint8).astype(np.uint8)
            return out

        # ------------------- validate 主流程 -------------------
        device = torch.device("cuda" if self.state.get("use_gpu", torch.cuda.is_available()) else "cpu")
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)

        all_probs = []
        all_targets = []

        cam_vis = bool(self.state.get("cam_vis", False))
        cam_first_batch_inp_cpu = None  # 缓存第一个 batch 的输入（CPU 版本）

        use_pb = bool(self.state.get('use_pb', True))
        pb_postfix_freq = max(1, int(self.state.get('pb_postfix_freq', 20)))
        if use_pb:
            data_loader = tqdm(
                data_loader,
                desc='Val',
                dynamic_ncols=True,
                leave=True,
                mininterval=float(self.state.get('pb_mininterval', 1.0)),
                miniters=int(self.state.get('pb_miniters', 10)),
                position=0,
            )

        end = time.time()
        for i, (inp, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            # 缓存第一个 batch（用于 CAM 可视化）
            if cam_vis and i == 0:
                cam_first_batch_inp_cpu = inp

            self.state['input'] = inp
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            # ---- 搬到设备 ----
            inp_dev = _to_device(inp, device)
            tgt_dev = target.to(device, non_blocking=True) if torch.is_tensor(target) else target

            # ---- 前向 + loss（验证用 no_grad）----
            with torch.no_grad():
                out = _call_model(model, inp_dev)
                out_logits = out[0] if isinstance(out, (tuple, list)) else out
                loss = criterion(out_logits, tgt_dev)

            # ---- 写回 state / meters ----
            self.state['output'] = out_logits
            self.state['loss'] = loss

            # ---- 收集 prob/target（算 AUC）----
            prob = torch.sigmoid(out_logits).detach().cpu()
            all_probs.append(prob)

            if torch.is_tensor(target):
                all_targets.append(target.detach().cpu())
            else:
                # 极少见：target 不是 tensor
                all_targets.append(torch.as_tensor(target))

            # ---- time / batch end ----
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(False, model, criterion, data_loader)

            if use_pb and (((i + 1) % pb_postfix_freq == 0) or ((i + 1) == len(data_loader))):
                avg_loss = self.state['meter_loss'].value()[0]
                cur_loss = float(self.state.get('loss_batch', 0.0))
                pct = 100.0 * float(i + 1) / max(1.0, float(len(data_loader)))
                data_loader.set_postfix_str(
                    f'{pct:5.1f}% | loss {cur_loss:.4f} | avg {avg_loss:.4f}'
                )

        # 打印平均 loss
        self.on_end_epoch(False, model, criterion, data_loader)

        # ------------------- 计算 AUC / mAP -------------------
        if len(all_probs) > 0:
            prob = torch.cat(all_probs, dim=0).numpy()
        else:
            prob = np.zeros((0, 1), dtype=np.float32)

        if len(all_targets) > 0:
            tgt = torch.cat(all_targets, dim=0).numpy()
        else:
            tgt = np.zeros((0, 1), dtype=np.int64)

        try:
            if prob.ndim == 1 or prob.shape[1] == 1:
                auc_macro = roc_auc_score(tgt.reshape(-1), prob.reshape(-1))
                mAP_score = average_precision_score(tgt.reshape(-1), prob.reshape(-1))
            else:
                auc_macro = roc_auc_score(tgt, prob, average='macro')
                mAP_score = average_precision_score(tgt, prob, average='macro')
        except Exception as e:
            print(f"计算 AUC/mAP 时出错: {e}")
            auc_macro, mAP_score = 0.5, 0.5

        print(f"Test Metrics: Macro-AUC={auc_macro:.4f}  mAP={mAP_score:.4f}")

        # ------------------- 逐类 AUROC -------------------
        per_class_auc = None
        try:
            C = prob.shape[1] if prob.ndim == 2 else 1
            per_class_auc = np.full(C, np.nan, dtype=np.float32)
            for j in range(C):
                y = tgt[:, j] if tgt.ndim == 2 else tgt.reshape(-1)
                p = prob[:, j] if prob.ndim == 2 else prob.reshape(-1)
                if np.max(y) == np.min(y):  # 全正或全负，ROC-AUC 不定义
                    continue
                per_class_auc[j] = roc_auc_score(y, p)
        except Exception as e:
            print(f"逐类 AUC 计算时出错: {e}")
            per_class_auc = None

        if per_class_auc is not None:
            print('Per-label AUROC:')
            label_names = self.state.get('label_names', None)
            for j in range(len(per_class_auc)):
                name = label_names[j] if (isinstance(label_names, (list, tuple)) and j < len(label_names)) else f'Label_{j}'
                aucj = per_class_auc[j]
                if np.isnan(aucj):
                    print(f'  {name:<20s}: N/A (no pos/neg)')
                else:
                    print(f'  {name:<20s}: {aucj:.4f}')

        # ------------------- 可选：导出 CAM 网格图（第一个 batch）-------------------
        if cam_vis and (cam_first_batch_inp_cpu is not None):
            try:
                core = _unwrap(model).eval()
                # 选择 CAM 目标层：默认用 ms_fuse（你的模型中确实存在且输出为 4D）；
                # 你也可以在 state 里指定 cam_target_layer 为其他模块名，如 "ms_fuse.out_norm"
                layer_name = self.state.get("cam_target_layer", "ms_fuse")

                # 兼容 "a.b.c" 这种层名
                target_module = core
                for part in str(layer_name).split("."):
                    target_module = getattr(target_module, part)

                cam_engine = _GradCAM2D(core, target_module)

                # 准备输入（GPU）
                batch_inp = _to_device(cam_first_batch_inp_cpu, device)

                # 获取 logits 以确定类别数
                with torch.no_grad():
                    out = _call_model(core, batch_inp)
                    out_logits = out[0] if isinstance(out, (tuple, list)) else out
                    num_classes = out_logits.shape[1]

                # 画哪些类：默认全部；也可用 cam_class_ids / cam_topk 控制
                class_ids = self.state.get("cam_class_ids", None)
                if class_ids is None:
                    # 可选：只画 topk
                    topk = self.state.get("cam_topk", None)
                    if topk is not None:
                        probs0 = torch.sigmoid(out_logits[0]).detach().cpu()
                        topk_ids = torch.topk(probs0, k=min(int(topk), num_classes)).indices.tolist()
                        class_ids = topk_ids
                    else:
                        class_ids = list(range(num_classes))

                # 取输入图像并反归一化做底图（对齐模型内部 resize）
                images = _extract_images(batch_inp)
                if hasattr(core, "input_size"):
                    images = F.interpolate(images, size=(core.input_size, core.input_size),
                                        mode="bicubic", align_corners=False)

                mean = getattr(core, "image_normalization_mean", [0.485, 0.456, 0.406])
                std  = getattr(core, "image_normalization_std",  [0.229, 0.224, 0.225])
                rgb_uint8 = _denorm_to_uint8(images, mean, std)[0].detach().cpu().numpy()  # (H,W,3)

                # 排版与保存
                cols = int(self.state.get("cam_cols", 4))
                alpha = float(self.state.get("cam_alpha", 0.45))
                rows = int(np.ceil(len(class_ids) / cols))

                label_names = self.state.get("label_names", None)
                if not (isinstance(label_names, (list, tuple)) and len(label_names) >= num_classes):
                    label_names = [f"Label_{i}" for i in range(num_classes)]

                save_dir = self.state.get("cam_save_dir", "./cam_out")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"val_epoch{self.state.get('epoch', 0)}_cam.png")

                plt.figure(figsize=(4 * cols, 4 * rows))
                for k, cid in enumerate(class_ids):
                    cam_map = cam_engine.cam_for_class(batch_inp, class_idx=int(cid), sample_idx=0)
                    overlay = _overlay_cam(rgb_uint8, cam_map, alpha=alpha)

                    ax = plt.subplot(rows, cols, k + 1)
                    ax.imshow(overlay)
                    ax.set_title(str(label_names[int(cid)]), fontsize=12)
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"[CAM] saved -> {save_path}")

            except Exception as e:
                print(f"[CAM] failed: {e}")

        # 返回 Macro-AUC 作为保存最佳的依据
        return float(auc_macro)


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if bool(self.state.get('disable_checkpoint', False)):
            best = state.get('best_score', self.state.get('best_score', None))
            msg = "[skip] checkpoint saving disabled"
            if best is not None:
                msg += f" (current best AUC = {best:.4f})"
            print(msg)
            return

        save_dir = self._state('save_model_path') or './checkpoint'
        os.makedirs(save_dir, exist_ok=True)

        save_best_only = bool(self.state.get('save_best_only', True))
        if (not save_best_only) or bool(is_best):
            ckpt_path = os.path.join(save_dir, filename)
            torch.save(state, ckpt_path)
            if is_best:
                best_path = os.path.join(save_dir, 'model_best.pth.tar')
                shutil.copyfile(ckpt_path, best_path)
            if self.state.get('epoch', None) is not None:
                print(f"[checkpoint] saved epoch={self.state['epoch']} best={bool(is_best)}")


    def adjust_learning_rate(self, optimizer):
        """
        Supports:
          - step decay: lr = base_lr * (lr_decay ** num_hits)
          - cosine decay: lr = base_lr * ratio(epoch)
          - optional linear warmup in early epochs
        """
        epoch = int(self.state.get('epoch', 0))
        scheduler = str(self.state.get('lr_scheduler', 'step')).lower()
        lr_decay = float(self.state.get('lr_decay', 0.2))
        warmup_epochs = max(0, int(self.state.get('warmup_epochs', 0)))
        min_lr_ratio = float(self.state.get('min_lr_ratio', 0.05))
        max_epochs = max(1, int(self.state.get('max_epochs', 1)))

        if '_base_lrs' not in self.state:
            self.state['_base_lrs'] = [float(pg.get('lr', 0.0)) for pg in optimizer.param_groups]
        base_lrs = self.state['_base_lrs']

        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_scale = float(epoch + 1) / float(warmup_epochs)
            lrs = [b * warmup_scale for b in base_lrs]
        else:
            if scheduler == 'cosine':
                t = max(0, epoch - warmup_epochs)
                T = max(1, max_epochs - warmup_epochs)
                progress = min(1.0, float(t) / float(T))
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                ratio = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                lrs = [b * ratio for b in base_lrs]
            else:
                steps = [int(x) for x in self.state.get('epoch_step', [])]
                num_hits = sum(1 for s in steps if epoch >= s)
                lrs = [b * (lr_decay ** num_hits) for b in base_lrs]

        for pg, lr in zip(optimizer.param_groups, lrs):
            pg['lr'] = float(lr)
        return np.unique(np.array(lrs, dtype=np.float32))







class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        # compute output
        self.state['output'] = model(feature_var, inp_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            accum_steps = max(1, int(self.state.get('accum_steps', 1)))
            cur_iter = int(self.state.get('iteration', 0))
            total_iters = int(self.state.get('num_steps_epoch', 0))

            if (cur_iter % accum_steps) == 0:
                optimizer.zero_grad(set_to_none=True)

            raw_loss = self.state['loss']
            (raw_loss / float(accum_steps)).backward()

            should_step = ((cur_iter + 1) % accum_steps == 0) or ((cur_iter + 1) == total_iters)
            if should_step:
                grad_clip = float(self.state.get('grad_clip', 10.0))
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            # keep the unscaled loss for logger/meter
            self.state['loss'] = raw_loss


    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]
