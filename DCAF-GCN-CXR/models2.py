# models2.py
# -*- coding: utf-8 -*-

import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
from torch.nn import Parameter
from torchvision.models import DenseNet121_Weights
from sklearn.metrics.pairwise import cosine_similarity

from util import *
from layers import *

# OverLoCK / ContMix 妯″潡锛堜綘鍙互鎶婃ā鍧?鍗曠嫭鏀惧埌 contmix.py 涓級
from contmix import ContMix  # <<< 鏂板锛氫粠浣犵殑 OverLoCK 妯″潡鏂囦欢涓鍏?ContMix


# =========================
# 宸ュ叿锛歱os_embed 鎻掑€硷紙甯?涓嶅甫 cls_token 鍧囧彲锛?# =========================
def interpolate_pos_embed(vit, pos_embed: torch.Tensor) -> torch.Tensor:
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)  # (1, N, C)

    # estimate ViT token grid from patch_embed when available
    if hasattr(vit, "patch_embed") and hasattr(vit.patch_embed, "grid_size"):
        gh, gw = map(int, vit.patch_embed.grid_size)
    else:
        num_patches = getattr(vit.patch_embed, "num_patches", None)
        if num_patches is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                toks = vit.forward_features(dummy)
                P = toks.shape[1] - 1 if toks.dim() == 3 else 1
        else:
            P = num_patches
        g = int(P ** 0.5)
        gh = gw = max(g, 1)

    # 鎷嗗垎 cls/grid
    num_tokens = pos_embed.shape[1]
    has_cls = 1 if num_tokens != gh * gw else 0
    if has_cls:
        cls_tok = pos_embed[:, :1, :]
        pos_grid = pos_embed[:, 1:, :]
    else:
        cls_tok, pos_grid = None, pos_embed

    gs_old = int(math.sqrt(pos_grid.shape[1]))
    pos_grid = pos_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # (1, C, H, W)
    pos_grid = F.interpolate(pos_grid, size=(gh, gw), mode="bicubic", align_corners=False)
    pos_grid = pos_grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)  # (1, P', C)

    if has_cls and cls_tok is not None:
        pos_embed_new = torch.cat([cls_tok, pos_grid], dim=1)
    else:
        pos_embed_new = pos_grid
    return pos_embed_new


def load_mae_to_vit(vit: torch.nn.Module, ckpt_path: str):
    """
    灏?MAE 棰勮缁冩潈閲嶅姞杞藉埌 timm 鐨?ViT 妯″瀷涓婏紙鍏煎 PyTorch 2.6+ 瀹夊叏鍙嶅簭鍒楀寲锛夈€?    """
    ckpt = None
    # 1) 棣栭€夛細鍏佽 argparse.Namespace 浣滀负瀹夊叏鍏ㄥ眬
    try:
        from torch.serialization import safe_globals
        from argparse import Namespace
        with safe_globals([Namespace]):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception as e1:
        print(f"[MAE] safe load with weights_only=True failed: {e1}")
        # 2) compatibility fallback for older torch versions
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        except TypeError:
            # 3) 鍏滃簳锛氭樉寮忓叧闂?weights_only
            print("[MAE] fallback to weights_only=False (make sure checkpoint is from a trusted source).")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 鎻愬彇 state_dict
    if isinstance(ckpt, dict):
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state = getattr(ckpt, 'state_dict', lambda: ckpt)()
        if callable(state):
            state = state()

    # 1) drop unrelated decoder / head weights
    drop_prefix = ('decoder', 'mask_token', 'head')
    state = {k: v for k, v in state.items() if not k.startswith(drop_prefix)}

    # 2) 鍘绘帀娼滃湪鐨?'encoder.' 鍓嶇紑
    if any(k.startswith('encoder.') for k in state):
        state = {k.replace('encoder.', ''): v for k, v in state.items()}

    # 3) interpolate positional embedding when resolution differs
    if 'pos_embed' in state and hasattr(vit, 'pos_embed') and vit.pos_embed.shape != state['pos_embed'].shape:
        try:
            state['pos_embed'] = interpolate_pos_embed(vit, state['pos_embed'])
        except Exception as e:
            print(f'[MAE] interpolate pos_embed failed: {e}')

    # 4) 娉ㄥ叆
    missing, unexpected = vit.load_state_dict(state, strict=False)
    print(f'[MAE] loaded -> missing: {len(missing)}, unexpected: {len(unexpected)}')
    if len(missing) > 0:
        print(f'[MAE] missing keys: {missing[:8]}{" ..." if len(missing) > 8 else ""}')
    if len(unexpected) > 0:
        print(f'[MAE] unexpected keys: {unexpected[:8]}{" ..." if len(unexpected) > 8 else ""}')
    return missing, unexpected


def load_densenet121_from_checkpoint(dn: nn.Module, ckpt_path: str):
    """
    灏?torchvision 鐨?densenet121 鍔犺浇涓烘寚瀹?checkpoint 鏉冮噸銆?    """
    if not (isinstance(ckpt_path, str) and os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"[DenseNet] checkpoint not found: {ckpt_path}")

    ckpt = None
    try:
        from torch.serialization import safe_globals
        from argparse import Namespace
        with safe_globals([Namespace]):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except Exception:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    sd = ckpt.get('state_dict', ckpt)
    new_sd = {}
    for k, v in sd.items():
        k = k.replace('module.', '')
        if k.startswith('classifier.'):
            continue
        new_sd[k] = v

    missing, unexpected = dn.load_state_dict(new_sd, strict=False)
    print(f"[DenseNet] loaded -> missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"[DenseNet] missing (showing up to 8): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[DenseNet] unexpected (up to 8): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return missing, unexpected


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if adj.device != input.device or adj.dtype != input.dtype:
            adj = adj.to(device=input.device, dtype=input.dtype)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + ' ('
            + str(self.in_features)
            + ' -> '
            + str(self.out_features)
            + ')'
        )


# ------------------------------
# 宸ュ叿锛歱os_embed reshape 鎻掑€?# ------------------------------
def resize_pos_embed(pos_embed: torch.Tensor, new_hw, has_cls_token=True):
    """
    灏?ViT 鐨?pos_embed 鎻掑€煎埌鏂?token 缃戞牸灏哄 new_hw=(H_tokens, W_tokens)銆?    鏀寔甯?涓嶅甫 cls token 鐨勪袱绉嶅舰寮忋€?    """
    if has_cls_token:
        cls_pos = pos_embed[:, :1, :]  # (1, 1, C)
        grid_pos = pos_embed[:, 1:, :]  # (1, N, C)
    else:
        cls_pos, grid_pos = None, pos_embed

    B, N, C = grid_pos.shape
    H_old = W_old = int(round(N ** 0.5))
    grid_pos = grid_pos.reshape(1, H_old, W_old, C).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(grid_pos, size=new_hw, mode='bicubic', align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], C)

    if cls_pos is not None:
        out = torch.cat([cls_pos, grid_pos], dim=1)
    else:
        out = grid_pos
    return out


# =========================
# GeM 姹犲寲
# =========================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):  # (B, C, H, W)
        x = torch.clamp(x, min=self.eps).pow(self.p).mean(dim=(2, 3))
        return x.pow(1.0 / self.p)


# =========================
# Mona2D锛氬灏哄害鍗风Н adapter锛?D锛?# =========================
class Mona2D(nn.Module):
    def __init__(self, in_ch, mid_ch=64, p_drop=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_ch)
        self.gamma = nn.Parameter(torch.ones(in_ch) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_ch))

        self.reduce = nn.Conv2d(in_ch, mid_ch, 1, bias=False)

        self.dw3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch, bias=False)
        self.dw5 = nn.Conv2d(mid_ch, mid_ch, 5, padding=2, groups=mid_ch, bias=False)
        self.dw7 = nn.Conv2d(mid_ch, mid_ch, 7, padding=3, groups=mid_ch, bias=False)

        self.proj = nn.Conv2d(mid_ch, in_ch, 1, bias=False)
        self.dropout = nn.Dropout2d(p=p_drop)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.norm(x) * self.gamma.view(1, -1, 1, 1) + x * self.gammax.view(1, -1, 1, 1)
        y = self.reduce(y)
        y = (self.dw3(y) + self.dw5(y) + self.dw7(y)) / 3.0 + y
        y = self.act(y)
        y = self.dropout(y)
        y = self.proj(y)
        return x + y


# =========================
# Dense 浣滀负 PatchEmbed锛堝彲閫夛級
# =========================
class DensePatchEmbed(nn.Module):
    def __init__(self, embed_dim=768, densenet_checkpoint_path=None,
                 pretrained_dense=True, proj_stride=4):
        super().__init__()
        if densenet_checkpoint_path:
            dn = models.densenet121(weights=None)
            load_densenet121_from_checkpoint(dn, densenet_checkpoint_path)
        else:
            dn = models.densenet121(
                weights=(
                    DenseNet121_Weights.IMAGENET1K_V1
                    if pretrained_dense else None
                )
            )

        feats = list(dn.features.children())
        self.stem = nn.Sequential(*feats[:5])  # (B, 256, H/4, W/4)

        self.proj = nn.Conv2d(256, embed_dim, proj_stride, proj_stride, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        f = self.stem(x)  # (B,256,H/4,W/4)
        x = self.proj(f)  # (B,embed_dim,H/16,W/16)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)
        return tokens, (H, W), x


# ------------------------------
# 宸ュ叿锛?脳1 鍗风Н
# ------------------------------
def conv1x1(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)


# =========================
# DCAF锛氬弻鍚戜氦鍙夋敞鎰?ViT鈫擟NN
# =========================
class DCAFusion(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0,
                 init_res_scale: float = 1.0):
        super().__init__()
        assert channels % num_heads == 0
        self.c = channels
        self.h = num_heads
        self.d = channels // num_heads
        self.scale = self.d ** -0.5

        self.ln_vit = nn.LayerNorm(channels)
        self.ln_cnn = nn.LayerNorm(channels)

        self.q_v = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_c = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_c = nn.Linear(channels, channels, bias=qkv_bias)

        self.q_c = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_v = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_v = nn.Linear(channels, channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_c = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.alpha_v = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))
        self.alpha_c = nn.Parameter(torch.tensor(init_res_scale, dtype=torch.float32))

    def _mh(self, x, B, N):
        return x.view(B, N, self.h, self.d).transpose(1, 2).contiguous()

    def forward(self, vit_2d: torch.Tensor, cnn_2d: torch.Tensor, mode: str = "bidirectional"):
        """
        mode:
          - bidirectional: ViT<->CNN both directions
          - v2c: only update CNN branch with ViT->CNN attention
          - c2v: only update ViT branch with CNN->ViT attention
        """
        B, C, H, W = vit_2d.shape
        N = H * W

        vit_seq = vit_2d.flatten(2).transpose(1, 2)  # (B, N, C)
        cnn_seq = cnn_2d.flatten(2).transpose(1, 2)

        v_norm = self.ln_vit(vit_seq)
        c_norm = self.ln_cnn(cnn_seq)

        mode = str(mode).lower()
        do_v2c = mode in ("bidirectional", "v2c")
        do_c2v = mode in ("bidirectional", "c2v")
        if not (do_v2c or do_c2v):
            raise ValueError(f"Unsupported DCAF mode: {mode}")

        v_fused = vit_2d
        c_fused = cnn_2d

        if do_v2c:
            qv = self._mh(self.q_v(v_norm), B, N)
            kc = self._mh(self.k_c(c_norm), B, N)
            vc = self._mh(self.v_c(c_norm), B, N)
            attn_v2c = (qv @ kc.transpose(-2, -1)) * self.scale
            attn_v2c = self.attn_drop(attn_v2c.softmax(dim=-1))
            out_v2c = attn_v2c @ vc
            out_v2c = out_v2c.transpose(1, 2).reshape(B, N, C)
            v2c_2d = out_v2c.transpose(1, 2).reshape(B, C, H, W)
            c_fused = c_fused + self.alpha_c * self.proj_drop(self.proj_c(v2c_2d))

        if do_c2v:
            qc = self._mh(self.q_c(c_norm), B, N)
            kv = self._mh(self.k_v(v_norm), B, N)
            vv = self._mh(self.v_v(v_norm), B, N)
            attn_c2v = (qc @ kv.transpose(-2, -1)) * self.scale
            attn_c2v = self.attn_drop(attn_c2v.softmax(dim=-1))
            out_c2v = attn_c2v @ vv
            out_c2v = out_c2v.transpose(1, 2).reshape(B, N, C)
            c2v_2d = out_c2v.transpose(1, 2).reshape(B, C, H, W)
            v_fused = v_fused + self.alpha_v * self.proj_drop(self.proj_v(c2v_2d))

        return v_fused, c_fused


# =========================
# DenseNet121 澶氬昂搴︼紙鎺ュ叆 OverLoCK / ContMix锛?# =========================
class DenseNet121MultiScale(nn.Module):
    """
    Expose DenseNet intermediate maps for flexible scale ablations.
    Returned keys: c1, c2, c3, c4.
    """
    OUT_CHANNELS = {"c1": 256, "c2": 512, "c3": 1024, "c4": 1024}

    def __init__(self, densenet_features: nn.Sequential, use_contmix: bool = False):
        super().__init__()
        f = densenet_features
        self.stem = nn.Sequential(f[0], f[1], f[2], f[3])
        self.db1 = f[4]
        self.tr1 = f[5]
        self.db2 = f[6]
        self.tr2 = f[7]
        self.db3 = f[8]
        self.tr3 = f[9]
        self.db4 = f[10]
        self.norm5 = f[11]

        self.use_contmix = use_contmix
        if use_contmix:
            self.cm3 = ContMix(
                dim=1024,
                ctx_dim=512,
                kernel_size=7,
                smk_size=5,
                num_heads=4,
                deploy=False,
                use_gemm=False,
            )
            self.cm4 = ContMix(
                dim=1024,
                ctx_dim=512,
                kernel_size=7,
                smk_size=5,
                num_heads=4,
                deploy=False,
                use_gemm=False,
            )

    def forward(self, x):
        x = self.stem(x)

        c1 = self.db1(x)
        x = self.tr1(c1)

        c2 = self.db2(x)
        x = self.tr2(c2)

        c3 = self.db3(x)
        if self.use_contmix:
            c3 = self.cm3(c3)
        x = self.tr3(c3)

        c4 = self.norm5(self.db4(x))
        if self.use_contmix:
            c4 = self.cm4(c4)

        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4}

class SpatialGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = conv1x1(ch, 1, bias=True)

    def forward(self, x, skip=None):
        g = torch.sigmoid(self.conv(x))  # (B,1,H,W)
        if skip is None:
            return x * g
        return x * g + skip * (1.0 - g)


# =========================
# 澶氬昂搴?DCAF + FPN 寮忚瀺鍚堬紙涓嶅甫 Mona锛?# =========================
class MultiScaleDCAF(nn.Module):
    """
    Configurable multi-scale fusion module for fine-grained ablations.
    """
    VALID_SCALES = ("c1", "c2", "c3", "c4")
    SCALE_CHANNELS = {"c1": 256, "c2": 512, "c3": 1024, "c4": 1024}
    VALID_FUSION = ("dcaf_bi", "dcaf_v2c", "dcaf_c2v", "concat", "weighted_sum")

    def __init__(
        self,
        ch=1024,
        heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        init_res_scale=1.0,
        scales=("c3", "c4"),
        fusion_type="dcaf_bi",
    ):
        super().__init__()
        scales = [str(s).lower() for s in scales]
        if len(scales) == 0:
            raise ValueError("At least one fusion scale must be provided.")
        for s in scales:
            if s not in self.VALID_SCALES:
                raise ValueError(f"Unsupported fusion scale: {s}")

        fusion_type = str(fusion_type).lower()
        if fusion_type not in self.VALID_FUSION:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.ch = int(ch)
        self.scales = tuple(scales)
        self.fusion_type = fusion_type
        self.dcaf_mode = {
            "dcaf_bi": "bidirectional",
            "dcaf_v2c": "v2c",
            "dcaf_c2v": "c2v",
        }.get(fusion_type, None)

        self.reducers = nn.ModuleDict()
        self.dcaf_blocks = nn.ModuleDict()
        self.intra_fuse = nn.ModuleDict()
        self.weight_logits = nn.ParameterDict()

        for s in self.scales:
            in_ch = self.SCALE_CHANNELS[s]
            self.reducers[s] = conv1x1(in_ch, self.ch, bias=False)

            if self.fusion_type in ("dcaf_bi", "dcaf_v2c", "dcaf_c2v"):
                self.dcaf_blocks[s] = DCAFusion(
                    self.ch,
                    num_heads=heads,
                    qkv_bias=qkv_bias,
                    attn_dropout=attn_drop,
                    proj_dropout=proj_drop,
                    init_res_scale=init_res_scale,
                )
                self.intra_fuse[s] = conv1x1(self.ch * 2, self.ch, bias=False)
            elif self.fusion_type == "concat":
                self.intra_fuse[s] = conv1x1(self.ch * 2, self.ch, bias=False)
            elif self.fusion_type == "weighted_sum":
                self.weight_logits[s] = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        if len(self.scales) > 1:
            self.scale_fuse = conv1x1(self.ch * len(self.scales), self.ch, bias=False)
        else:
            self.scale_fuse = nn.Identity()

        self.spatial_gate = SpatialGate(self.ch)
        self.out_norm = nn.BatchNorm2d(self.ch)

    def _fuse_one_scale(self, scale_name, vit_2d, dense_2d):
        if self.fusion_type in ("dcaf_bi", "dcaf_v2c", "dcaf_c2v"):
            v_fused, c_fused = self.dcaf_blocks[scale_name](
                vit_2d, dense_2d, mode=self.dcaf_mode
            )
            return self.intra_fuse[scale_name](torch.cat([v_fused, c_fused], dim=1))

        if self.fusion_type == "concat":
            return self.intra_fuse[scale_name](torch.cat([vit_2d, dense_2d], dim=1))

        if self.fusion_type == "weighted_sum":
            w = torch.softmax(self.weight_logits[scale_name], dim=0)
            return w[0] * vit_2d + w[1] * dense_2d

        raise RuntimeError(f"Unexpected fusion_type: {self.fusion_type}")

    def forward(self, vit_2d, dense_feats, out_hw):
        H, W = out_hw
        per_scale = []

        for s in self.scales:
            if s not in dense_feats:
                raise KeyError(f"Dense feature map '{s}' not found. Available={list(dense_feats.keys())}")
            feat = self.reducers[s](dense_feats[s])
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
            per_scale.append(self._fuse_one_scale(s, vit_2d, feat))

        if len(per_scale) == 1:
            fused = per_scale[0]
        else:
            fused = self.scale_fuse(torch.cat(per_scale, dim=1))

        fused = self.spatial_gate(fused, skip=vit_2d)
        return self.out_norm(fused)

class ForegroundAttentionBlock(nn.Module):
    """
    CBAM 椋庢牸 FAB锛氶€氶亾娉ㄦ剰 (avg+max) + 绌洪棿娉ㄦ剰 (avg&max map 鎷兼帴)
    杈撳嚭鍓嶆櫙澧炲己鐗瑰緛 Qv(st) 鍙婂墠鏅?mask
    """
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_ch // r, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        xc = x * ca

        # Spatial attention
        avg_map = torch.mean(xc, dim=1, keepdim=True)
        max_map, _ = torch.max(xc, dim=1, keepdim=True)
        sa_in = torch.cat([avg_map, max_map], dim=1)  # (B,2,H,W)
        m = torch.sigmoid(self.spatial(sa_in))        # (B,1,H,W)
        xf = xc * m

        return xf, m


# =========================
# 璇婃柇浠ｇ悊锛氭爣绛炬劅鐭?Transformer 澶?# =========================
class DiagnosticTransformerHead(nn.Module):
    """
    璇婃柇 agent Q_d: 鏍囩浣滀负 query, 鍏堥獙鐗瑰緛 (Qv, Qs) 浣滀负 K/V
    甯?LayerNorm + 娈嬪樊锛岃缁冩洿绋?    """
    def __init__(self, num_classes, d_model=1024,
                 num_layers=2, num_heads=4,
                 dim_ff=2048, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, 1)
        self.num_classes = num_classes

    def forward(self, f_vis_2d, sem_vec, label_emb):
        """
        f_vis_2d: (B, C, H, W)  瑙嗚鍏堥獙 Qv(st)
        sem_vec:  (B, C)        璇箟鍏堥獙 Qs(st)
        label_emb: (K, C)       GCN 鏍囩宓屽叆, 浣滀负鍒濆 Q0

        return: logits (B, K)
        """
        B, C, H, W = f_vis_2d.shape
        K = label_emb.size(0)

        # K,V: 瑙嗚 tokens + 涓€涓涔?token
        src_spatial = f_vis_2d.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        sem_token = sem_vec.unsqueeze(1)                                # (B, 1, C)
        src = torch.cat([sem_token, src_spatial], dim=1)                # (B, 1+N, C)
        src = self.src_norm(src)

        # Q: label_emb 浣滀负 query 鍒濆€?Q0
        tgt0 = label_emb.unsqueeze(0).expand(B, K, C).contiguous()      # (B, K, C)
        tgt_norm = self.tgt_norm(tgt0)

        out = self.decoder(tgt_norm, src)                               # (B, K, C)
        out = self.out_norm(out + tgt0)                                 # 娈嬪樊淇濈暀 label 鍏堥獙

        logits = self.classifier(out).squeeze(-1)                       # (B, K)
        return logits


# =========================
# MonaTokenAdapter & MonaViTBlock锛氬湪 ViT Block 鍐呮彃 Mona
# =========================
class MonaTokenAdapter(nn.Module):
    """
    鎶?ViT token (B,N,C) reshape 鎴?2D锛岃窇 Mona2D锛屽啀杩樺師
    """
    def __init__(self, embed_dim, grid_size, mid_ch=64, p_drop=0.1):
        super().__init__()
        self.H, self.W = grid_size
        self.mona2d = Mona2D(in_ch=embed_dim, mid_ch=mid_ch, p_drop=p_drop)

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.H, self.W

        if N == H * W + 1:
            cls, tokens = x[:, :1, :], x[:, 1:, :]
        else:
            cls, tokens = None, x

        feat = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        feat = self.mona2d(feat)
        tokens_out = feat.flatten(2).transpose(1, 2).contiguous()

        if cls is not None:
            return torch.cat([cls, tokens_out], dim=1)
        else:
            return tokens_out


class MonaViTBlock(nn.Module):
    """
    timm ViT Block 鍖呰锛?      x = x + Attn(...)
      x = Mona_after_attn(x)
      x = x + MLP(...)
      x = Mona_after_mlp(x)
    """
    def __init__(self, base_block, grid_size, embed_dim, mid_ch=64, p_drop=0.1):
        super().__init__()
        self.b = base_block
        self.mona_after_attn = MonaTokenAdapter(embed_dim, grid_size, mid_ch, p_drop)
        self.mona_after_mlp = MonaTokenAdapter(embed_dim, grid_size, mid_ch, p_drop)

    def forward(self, x):
        b = self.b

        # MSA 瀛愬眰 + Mona
        shortcut = x
        x = b.norm1(x)
        x = b.attn(x)
        if hasattr(b, "drop_path") and b.drop_path is not None:
            x = b.drop_path(x)
        x = shortcut + x
        x = self.mona_after_attn(x)

        # MLP 瀛愬眰 + Mona
        shortcut = x
        x = b.norm2(x)
        x = b.mlp(x)
        if hasattr(b, "drop_path") and b.drop_path is not None:
            x = b.drop_path(x)
        x = shortcut + x
        x = self.mona_after_mlp(x)

        return x


# =========================
# 涓绘ā鍨嬶細ViT + Dense 澶氬昂搴?+ DCAF + MonaViT + 涓?agent + GCN
# =========================
def _load_label_embeddings_from_pickle(path, num_classes):
    if path is None:
        return None
    obj = pickle.load(open(path, "rb"))

    if isinstance(obj, dict):
        if "embeddings" in obj:
            arr = np.asarray(obj["embeddings"], dtype=np.float32)
        elif "inp" in obj:
            arr = np.asarray(obj["inp"], dtype=np.float32)
        else:
            # fallback: if dict is already index->vector
            vals = list(obj.values())
            arr = np.asarray(vals, dtype=np.float32)
    else:
        arr = np.asarray(obj, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"label embedding file must contain 2D matrix, got shape={arr.shape}")

    if arr.shape[0] != num_classes:
        if arr.shape[1] == num_classes:
            arr = arr.T
        else:
            raise ValueError(
                f"label embedding row count mismatch: expected {num_classes}, got {arr.shape}"
            )
    return np.asarray(arr, dtype=np.float32)

class ViT_Hybrid(nn.Module):
    def __init__(self, num_classes, in_channel=300, t=None, adj_file=None,
                 checkpoint_path=None, mae_ckpt_path=None,
                 model_name='vit_base_patch16_384', img_size=384,
                 densenet_checkpoint_path=None, pretrained_dense=True,
                 use_dense_as_patch: bool = False,
                 freeze_vit: bool = True,
                 unfreeze_last_n_blocks: int = 0,
                 learnable_adj: bool = True,
                 use_dense_overlock: bool = False,
                 fusion_type: str = "dcaf_bi",
                 fusion_scales=("c3", "c4"),
                 graph_source: str = "coocc_only",
                 graph_alpha: float = 0.5,
                 graph_edge_policy: str = "threshold",
                 graph_topk: int = 0,
                 label_emb_path: str = None):
        super().__init__()
        self.use_dense_as_patch = use_dense_as_patch
        self.learnable_adj = learnable_adj
        self.num_classes = num_classes
        self.fusion_type = str(fusion_type).lower()
        self.fusion_scales = tuple(str(s).lower() for s in fusion_scales)
        self.graph_source = str(graph_source).lower()
        self.graph_alpha = float(graph_alpha)
        self.graph_edge_policy = str(graph_edge_policy).lower()
        self.graph_topk = int(graph_topk)

        # -------- 1) ViT 涓诲共 --------
        self.input_size = int(img_size)
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=False if mae_ckpt_path else True,
                checkpoint_path=checkpoint_path,
                img_size=self.input_size,
            )
        except TypeError:
            self.vit = timm.create_model(
                model_name,
                pretrained=False if mae_ckpt_path else True,
                checkpoint_path=checkpoint_path,
            )

        self.embed_dim = getattr(self.vit, 'num_features',
                                 getattr(self.vit, 'embed_dim', 768))

        if mae_ckpt_path:
            load_mae_to_vit(self.vit, mae_ckpt_path)

        dcfg = getattr(self.vit, 'default_cfg', {}) or {}
        self.image_normalization_mean = dcfg.get('mean', [0.485, 0.456, 0.406])
        self.image_normalization_std = dcfg.get('std', [0.229, 0.224, 0.225])

        # patch_size & grid_size
        vit_ps = 16
        if hasattr(self.vit, 'patch_embed') and hasattr(self.vit.patch_embed, 'patch_size'):
            ps = self.vit.patch_embed.patch_size
            vit_ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        self.patch_size = vit_ps

        gh = self.input_size // self.patch_size
        gw = self.input_size // self.patch_size
        self.vit_grid = (gh, gw)

        # 鍏堟寜 Mona 鎬濊矾鍐荤粨
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # 鎸?2306 鎬濇兂锛氬彲閫夊湴瑙ｅ喕 backbone 灏鹃儴锛堟寔缁洿鏂帮級
        if unfreeze_last_n_blocks > 0:
            for blk in self.vit.blocks[-unfreeze_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # 鐢?MonaViTBlock 鏇挎崲鍘?blocks
        new_blocks = []
        for blk in self.vit.blocks:
            new_blocks.append(
                MonaViTBlock(
                    base_block=blk,
                    grid_size=self.vit_grid,
                    embed_dim=self.embed_dim,
                    mid_ch=64,
                    p_drop=0.1,
                )
            )
        self.vit.blocks = nn.ModuleList(new_blocks)

        # -------- 2) Dense 澶氬昂搴﹀垎鏀?--------
        if densenet_checkpoint_path:
            dn = models.densenet121(weights=None)
            load_densenet121_from_checkpoint(dn, densenet_checkpoint_path)
        else:
            dn = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # 杩欓噷鍚敤/鍏抽棴 OverLoCK-ContMix
        self.densenet = DenseNet121MultiScale(
            dn.features,
            use_contmix=use_dense_overlock,  # <<< 鏂板锛氬皢寮€鍏充紶缁欏灏哄害 DenseNet
        )
        self.dense_dim = 1024

        # 鍙€夛細Dense 浣滀负 patch_embed
        if self.use_dense_as_patch:
            assert vit_ps % 4 == 0
            self.dense_patch = DensePatchEmbed(
                embed_dim=self.embed_dim,
                densenet_checkpoint_path=densenet_checkpoint_path,
                pretrained_dense=pretrained_dense,
            )

        # -------- 3) ViT鈫?024 瀵归綈 + 澶氬昂搴﹁瀺鍚?--------
        self.proj_vit = nn.Conv2d(self.embed_dim, self.dense_dim, 1)
        nn.init.kaiming_normal_(self.proj_vit.weight, mode="fan_out",
                                nonlinearity="relu")

        self.ms_fuse = MultiScaleDCAF(
            ch=self.dense_dim, heads=8, qkv_bias=True,
            attn_drop=0.0, proj_drop=0.0, init_res_scale=1.0,
            fusion_type=self.fusion_type,
            scales=self.fusion_scales,
        )

        # -------- 4) 涓変釜 agent 鐩稿叧妯″潡 --------
        # visual prior branch
        self.fab = ForegroundAttentionBlock(self.dense_dim)

        # semantic prior branch
        self.gem = GeM(p=3.0)
        self.semantic_fc = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 鏍囩 GCN 鍏堥獙
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, self.dense_dim)
        self.relu = nn.LeakyReLU(0.2)
        label_embeddings = None
        if self.graph_source in ("joint", "semantic_only"):
            label_embeddings = _load_label_embeddings_from_pickle(label_emb_path, num_classes)
        _adj = gen_A(
            num_classes,
            t,
            adj_file,
            label_embeddings=label_embeddings,
            graph_source=self.graph_source,
            semantic_alpha=self.graph_alpha,
            edge_policy=self.graph_edge_policy,
            topk=self.graph_topk,
        )
        self.A = Parameter(torch.from_numpy(_adj).float())

        # 璇婃柇 Transformer agent
        self.diag_head = DiagnosticTransformerHead(
            num_classes=num_classes,
            d_model=self.dense_dim,
            num_layers=2,
            num_heads=4,
            dim_ff=2048,
            dropout=0.1,
        )

        # -------- RL-style 涓?agent 澶?--------
        # 瑙嗚 agent锛氬墠鏅敞鎰忕壒寰?-> 鍏ㄥ眬鍚戦噺 -> logits
        self.vis_fc = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.vis_cls = nn.Linear(self.dense_dim, num_classes)

        # 璇箟 agent锛氳涔夊厛楠屽悜閲?-> logits
        self.sem_cls = nn.Linear(self.dense_dim, num_classes)

        self._last_logits_vis = None
        self._last_logits_sem = None
        self._aux_logits = None

    # ========= ViT token 鎻愬彇 =========
    def _vit_tokens_from_native(self, images):
        vit = self.vit
        x = vit.patch_embed(images)  # (B, N, C)
        B, N, C = x.shape
        if hasattr(vit, 'cls_token') and vit.cls_token is not None:
            cls = vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            pe = vit.pos_embed
            if x.size(1) == pe.size(1):
                x = x + pe
            else:
                gh = images.shape[-2] // self.patch_size
                gw = images.shape[-1] // self.patch_size
                cls_pos, patch_pos = pe[:, :1, :], pe[:, 1:, :]
                g0 = int(patch_pos.size(1) ** 0.5)
                patch_pos_2d = patch_pos.transpose(1, 2).contiguous().view(1, C, g0, g0)
                patch_pos_2d = F.interpolate(
                    patch_pos_2d, size=(gh, gw),
                    mode='bilinear', align_corners=False
                )
                patch_pos_new = patch_pos_2d.flatten(2).transpose(1, 2).contiguous()
                pos = torch.cat([cls_pos, patch_pos_new], dim=1) \
                    if vit.cls_token is not None else patch_pos_new
                x = x + pos

        if hasattr(vit, 'pos_drop'):
            x = vit.pos_drop(x)

        for blk in vit.blocks:
            x = blk(x)
        if hasattr(vit, 'norm') and vit.norm is not None:
            x = vit.norm(x)
        return x

    def _vit_tokens_from_dense(self, images):
        out = self.dense_patch(images)
        if isinstance(out, (list, tuple)) and len(out) == 3:
            tokens, (H, W), _ = out
        else:
            tokens, (H, W) = out
        vit = self.vit
        B = tokens.size(0)
        if hasattr(vit, 'cls_token') and vit.cls_token is not None:
            cls = vit.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, tokens], dim=1)
        else:
            x = tokens

        if hasattr(vit, 'pos_embed') and vit.pos_embed is not None:
            pe = vit.pos_embed
            if x.size(1) == pe.size(1):
                x = x + pe
            else:
                C = tokens.size(-1)
                cls_pos, patch_pos = pe[:, :1, :], pe[:, 1:, :]
                g0 = int(patch_pos.size(1) ** 0.5)
                patch_pos_2d = patch_pos.transpose(1, 2).contiguous().view(1, C, g0, g0)
                patch_pos_2d = F.interpolate(
                    patch_pos_2d, size=(H, W),
                    mode='bilinear', align_corners=False
                )
                patch_pos_new = patch_pos_2d.flatten(2).transpose(1, 2).contiguous()
                pos = torch.cat([cls_pos, patch_pos_new], dim=1) \
                    if vit.cls_token is not None else patch_pos_new
                x = x + pos

        if hasattr(vit, 'pos_drop'):
            x = vit.pos_drop(x)
        for blk in vit.blocks:
            x = blk(x)
        if hasattr(vit, 'norm') and vit.norm is not None:
            x = vit.norm(x)
        return x, (H, W)

    # ========= forward =========
    def forward(self, images, inp, return_all: bool = False):
        # 缁熶竴 resize
        images = F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode='bicubic',
            align_corners=False
        )

        # ViT tokens
        if self.use_dense_as_patch:
            vit_tokens, (H, W) = self._vit_tokens_from_dense(images)
        else:
            vit_tokens = self._vit_tokens_from_native(images)
            H = images.shape[-2] // self.patch_size
            W = images.shape[-1] // self.patch_size

        if vit_tokens.dim() == 3:
            if vit_tokens.size(1) == (1 + H * W):
                patch_tokens = vit_tokens[:, 1:, :]
            else:
                patch_tokens = vit_tokens
            vit_2d = patch_tokens.permute(0, 2, 1).contiguous().view(
                -1, self.embed_dim, H, W
            )
        else:
            vit_2d = vit_tokens.view(vit_tokens.size(0), self.embed_dim, 1, 1)

        # Dense multi-scale features
        dense_feats = self.densenet(images)

        # 閫氶亾瀵归綈 + 澶氬昂搴?DCAF 铻嶅悎
        vit_2d_proj = self.proj_vit(vit_2d)                      # (B,1024,H,W)
        fused = self.ms_fuse(vit_2d_proj, dense_feats, out_hw=(H, W))  # (B,1024,H,W)

        # 瑙嗚鍏堥獙 Q岬ワ細鍓嶆櫙娉ㄦ剰鍚庣殑 feature
        f_vis, fg_mask = self.fab(fused)                          # (B,1024,H,W),(B,1,H,W)

        # 瑙嗚 agent锛氬彧鐪嬪墠鏅壒寰佺殑鍒嗙被鍐崇瓥 Q_v(s_t, 路)
        vis_vec = self.gem(f_vis)                                 # (B,1024)
        vis_vec = self.vis_fc(vis_vec)                            # (B,1024)
        logits_vis = self.vis_cls(vis_vec)                        # (B,num_classes)

        # 璇箟鍏堥獙 Q鈧涳細鍏ㄥ眬 pooled + MLP
        sem_vec = self.gem(fused)                                 # (B,1024)
        sem_vec = self.semantic_fc(sem_vec)                       # (B,1024)

        # 璇箟 agent锛氬彧鐪嬭涔夊悜閲忕殑鍒嗙被鍐崇瓥 Q_s(s_t, 路)
        logits_sem = self.sem_cls(sem_vec)                        # (B,num_classes)

        # 鏍囩 GCN 鍏堥獙
        if self.learnable_adj:
            adj = gen_adj(self.A)                                 # 鍏佽姊害鍒?A
        else:
            adj = gen_adj(self.A).detach()

        z = self.gc1(inp[0], adj)
        z = self.relu(z)
        label_emb = self.gc2(z, adj)                              # (K,1024)

        # 璇婃柇 Transformer 澶达細铻嶅悎 Q岬?+ Q鈧?+ GCN 鏍囩鍏堥獙 鈫?鏈€缁?Q_d
        logits = self.diag_head(f_vis, sem_vec, label_emb)        # (B,num_classes)

        # 缂撳瓨杈呭姪 head 杈撳嚭锛屼緵 MultiHeadCriterion / AuxWithVisSemLoss 浣跨敤
        self._last_logits_vis = logits_vis
        self._last_logits_sem = logits_sem
        self._aux_logits = (logits_vis, logits_sem)

        # 榛樿淇濇寔鑰佹帴鍙ｏ紝鍙緭鍑鸿瘖鏂?agent 鐨?logits
        if return_all:
            # final, visual, semantic
            return logits, logits_vis, logits_sem

        return logits

    # ========= 浼樺寲鍣ㄥ垎缁勶細鍖哄垎 ViT base / Mona / 鍏朵粬 =========
    def _get_config_optim_legacy(self, lr, lrp):
        vit_base_params = []
        vit_mona_params = []

        for name, p in self.vit.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if "mona" in lname:
                vit_mona_params.append(p)
            else:
                vit_base_params.append(p)

        align_params = list(self.proj_vit.parameters())
        dense_ms_params = list(self.densenet.parameters()) + list(self.ms_fuse.parameters())
        head_params = (
            list(self.fab.parameters())
            + list(self.gem.parameters())
            + list(self.semantic_fc.parameters())
            + list(self.vis_fc.parameters())
            + list(self.vis_cls.parameters())
            + list(self.sem_cls.parameters())
            + list(self.gc1.parameters())
            + list(self.gc2.parameters())
            + list(self.diag_head.parameters())
        )

        if self.use_dense_as_patch:
            dense_ms_params += list(self.dense_patch.parameters())

        param_groups = []
        if len(vit_base_params) > 0:
            param_groups.append({
                'params': vit_base_params,
                'lr': lr * lrp,
                'weight_decay': 1e-4
            })
        if len(vit_mona_params) > 0:
            param_groups.append({
                'params': vit_mona_params,
                'lr': lr,              # Mona 杈冮珮 lr
                'weight_decay': 1e-4
            })

        param_groups.extend([
            {'params': align_params,    'lr': lr,       'weight_decay': 1e-4},
            {'params': dense_ms_params, 'lr': lr * 2.0, 'weight_decay': 1e-4},
            {'params': head_params,     'lr': lr,       'weight_decay': 0.0},
        ])
        return param_groups

    # New optimizer grouping API (kept backward compatible with old call sites)
    def get_config_optim(
        self,
        lr,
        lrp,
        wd_backbone=5e-4,
        wd_task=1e-4,
        wd_head=1e-4,
        mona_lr_mult=1.0,
        dense_lr_mult=2.0,
        align_lr_mult=1.0,
        head_lr_mult=1.0,
        ms_fuse_lr_mult=1.0,
    ):
        vit_base_params = []
        vit_mona_params = []

        for name, p in self.vit.named_parameters():
            if not p.requires_grad:
                continue
            if "mona" in name.lower():
                vit_mona_params.append(p)
            else:
                vit_base_params.append(p)

        densenet_params = [p for p in self.densenet.parameters() if p.requires_grad]
        ms_fuse_params = [p for p in self.ms_fuse.parameters() if p.requires_grad]
        if self.use_dense_as_patch:
            ms_fuse_params += [p for p in self.dense_patch.parameters() if p.requires_grad]
        align_params = [p for p in self.proj_vit.parameters() if p.requires_grad]

        head_params = (
            [p for p in self.fab.parameters() if p.requires_grad]
            + [p for p in self.gem.parameters() if p.requires_grad]
            + [p for p in self.semantic_fc.parameters() if p.requires_grad]
            + [p for p in self.vis_fc.parameters() if p.requires_grad]
            + [p for p in self.vis_cls.parameters() if p.requires_grad]
            + [p for p in self.sem_cls.parameters() if p.requires_grad]
            + [p for p in self.gc1.parameters() if p.requires_grad]
            + [p for p in self.gc2.parameters() if p.requires_grad]
            + [p for p in self.diag_head.parameters() if p.requires_grad]
        )

        param_groups = []
        if len(vit_base_params) > 0:
            param_groups.append({
                'name': 'vit_base',
                'params': vit_base_params,
                'lr': float(lr) * float(lrp),
                'weight_decay': float(wd_backbone),
            })
        if len(vit_mona_params) > 0:
            param_groups.append({
                'name': 'vit_mona',
                'params': vit_mona_params,
                'lr': float(lr) * float(mona_lr_mult),
                'weight_decay': float(wd_task),
            })
        if len(densenet_params) > 0:
            param_groups.append({
                'name': 'densenet',
                'params': densenet_params,
                'lr': float(lr) * float(dense_lr_mult),
                'weight_decay': float(wd_backbone),
            })
        if len(ms_fuse_params) > 0:
            param_groups.append({
                'name': 'ms_fuse',
                'params': ms_fuse_params,
                'lr': float(lr) * float(ms_fuse_lr_mult),
                'weight_decay': float(wd_task),
            })
        if len(align_params) > 0:
            param_groups.append({
                'name': 'align',
                'params': align_params,
                'lr': float(lr) * float(align_lr_mult),
                'weight_decay': float(wd_task),
            })
        if len(head_params) > 0:
            param_groups.append({
                'name': 'heads',
                'params': head_params,
                'lr': float(lr) * float(head_lr_mult),
                'weight_decay': float(wd_head),
            })
        return param_groups


# =========================
# 鏋勫缓鍑芥暟锛堟帴鍙ｅ熀鏈笉鍙橈級
# =========================
def gcn_SQNET_ViT(
    num_classes,
    t,
    pretrained=True,
    adj_file=None,
    checkpoint_path=None,
    in_channel=300,
    mae_ckpt_path=None,
    model_name="vit_base_patch16_384",
    img_size=384,
    densenet_checkpoint_path=None,
    pretrained_dense=True,
    use_dense_as_patch=False,
    freeze_vit=True,
    unfreeze_last_n_blocks=0,
    learnable_adj=True,
    use_dense_overlock=True,
    fusion_type="dcaf_bi",
    fusion_scales=("c3", "c4"),
    graph_source="coocc_only",
    graph_alpha=0.5,
    graph_edge_policy="threshold",
    graph_topk=0,
    label_emb_path=None,
):
    return ViT_Hybrid(
        num_classes,
        t=t,
        adj_file=adj_file,
        checkpoint_path=checkpoint_path,
        in_channel=in_channel,
        mae_ckpt_path=mae_ckpt_path,
        model_name=model_name,
        img_size=img_size,
        densenet_checkpoint_path=densenet_checkpoint_path,
        pretrained_dense=pretrained_dense,
        use_dense_as_patch=use_dense_as_patch,
        freeze_vit=freeze_vit,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        learnable_adj=learnable_adj,
        use_dense_overlock=use_dense_overlock,
        fusion_type=fusion_type,
        fusion_scales=fusion_scales,
        graph_source=graph_source,
        graph_alpha=graph_alpha,
        graph_edge_policy=graph_edge_policy,
        graph_topk=graph_topk,
        label_emb_path=label_emb_path,
    )

