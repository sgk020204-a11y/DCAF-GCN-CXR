# contmix.py
# OverLoCK: ContMix 模块 + DilatedReparamBlock
# - 去掉对 natten 的硬依赖，提供纯 PyTorch na2d_av() fallback
# - 兼容 Python 3.9（不使用 int | None 等新语法）

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum
from timm.models.layers import to_2tuple
from typing import Optional

# ================================================================
# 1. 可选导入 natten.na2d_av，如不可用则使用 PyTorch fallback
# ================================================================

try:
    from natten.functional import na2d_av as _natten_na2d_av
    _HAS_NATTEN = True
except Exception:
    _HAS_NATTEN = False
    _natten_na2d_av = None


def na2d_av(attn: torch.Tensor,
            value: torch.Tensor,
            kernel_size: int) -> torch.Tensor:
    """
    接口仿 natten.na2d_av(attn, value, kernel_size)
    形状：
        attn  : (B, Hheads, H, W, K*K)
        value : (B, Hheads, H, W, C)
    返回：
        out   : (B, Hheads, H, W, C)
    """

    if _HAS_NATTEN:
        return _natten_na2d_av(attn, value, kernel_size=kernel_size)

    # ---------- 纯 PyTorch fallback（会慢，但能跑） ----------
    # 思路：对 value 做 im2col / unfold 得到每个位置的 K*K 邻域特征，
    #       然后用 attn 作为权重做加权求和。
    B, Hh, H, W, K2 = attn.shape
    _, _, _, _, C = value.shape
    k = kernel_size
    assert K2 == k * k, f"attn 最后维度({K2}) 与 kernel_size^2 ({k*k}) 不一致"

    # value: (B, Hheads, H, W, C) -> (B*Hheads, C, H, W)
    v = value.permute(0, 1, 4, 2, 3).contiguous()
    v = v.view(B * Hh, C, H, W)

    # 展开邻域，得到每个位置 K*K patch
    patches = F.unfold(v, kernel_size=k, padding=k // 2, stride=1)  # (B*Hh, C*K2, H*W)
    patches = patches.view(B, Hh, C, K2, H, W)                      # (B,Hh,C,K2,H,W)

    # attn: (B, Hh, H, W, K2) -> (B,Hh,1,K2,H,W) 方便广播
    attn_ = attn.permute(0, 1, 4, 2, 3).contiguous().unsqueeze(2)  # (B,Hh,1,K2,H,W)

    # 加权求和：sum_k attn * patches
    out = (attn_ * patches).sum(dim=3)                             # (B,Hh,C,H,W)

    # 回到 (B,Hh,H,W,C)
    out = out.permute(0, 1, 3, 4, 2).contiguous()
    return out


# ================================================================
# 2. UniRepLKNet: Dilated Reparam Block 及相关工具
# ================================================================

def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl: bool = True):
    """
    统一封装 Conv2d，支持大核时尝试 iGEMM 深度卷积实现（可选）。
    """
    kernel_size = to_2tuple(kernel_size)

    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)

    need_large_impl = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM  # noqa
            print('---------------- found iGEMM implementation ')
        except Exception:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. '
                  'follow https://github.com/AILab-CVC/UniRepLKNet to install it.')

        if (
            DepthWiseConv2dImplicitGEMM is not None
            and in_channels == out_channels
            and out_channels == groups
            and stride == 1
            and dilation == 1
        ):
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, '
                  f'kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )


def get_bn(dim, use_sync_bn: bool = False):
    return nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)


def fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    conv + bn 融合为一个等效 conv
    """
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return w, b


def convert_dilated_to_nondilated(kernel: torch.Tensor, dilate_rate: int):
    """
    把一个「带 dilation 的 depthwise/group 卷积核」等效为更大尺寸的普通卷积核
    """
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device, dtype=kernel.dtype)

    if kernel.size(1) == 1:
        # DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        # group-wise 但非 DW
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel: torch.Tensor,
                                    dilated_kernel: torch.Tensor,
                                    dilated_r: int):
    """
    把多个 dilated 分支的等效大核叠加到原始大核上
    """
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block (UniRepLKNet)
    输入：(N, C, H, W)，输出同形状。
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 deploy: bool = False,
                 use_sync_bn: bool = False,
                 attempt_use_lk_impl: bool = True):
        super().__init__()

        self.lk_origin = get_conv2d(
            channels, channels, kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl
        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        # 默认多尺度 + 不同 dilation 的分支配置
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn=use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    f'dil_conv_k{k}_{r}',
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=k,
                        stride=1,
                        padding=(r * (k - 1) + 1) // 2,
                        dilation=r,
                        groups=channels,
                        bias=False
                    )
                )
                self.__setattr__(
                    f'dil_bn_k{k}_{r}',
                    get_bn(channels, use_sync_bn=use_sync_bn)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # deploy 模式：只有一个合并后的 lk_origin
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)

        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        """
        训练结束后可调用，将多分支融成一个大核 conv，用于推理加速。
        """
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f'dil_conv_k{k}_{r}')
                bn = self.__getattr__(f'dil_bn_k{k}_{r}')
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b

            merged_conv = get_conv2d(
                origin_k.size(0), origin_k.size(0), origin_k.size(2),
                stride=1,
                padding=origin_k.size(2) // 2,
                dilation=1,
                groups=origin_k.size(0),
                bias=True,
                attempt_use_lk_impl=self.attempt_use_lk_impl
            )
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b

            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f'dil_conv_k{k}_{r}')
                self.__delattr__(f'dil_bn_k{k}_{r}')


# ================================================================
# 3. ContMix 上下文混合动态卷积
# ================================================================

class ContMix(nn.Module):
    """
    ContMix: 上下文混合动态卷积
    输入 / 输出: (B, dim, H, W)
    """
    def __init__(self,
                 dim: int = 64,
                 ctx_dim: Optional[int] = None,
                 kernel_size: int = 7,
                 smk_size: int = 5,
                 num_heads: int = 2,
                 deploy: bool = False,
                 use_gemm: bool = False):
        super().__init__()

        if ctx_dim is None:
            ctx_dim = dim // 2

        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        # 局部编码 (LEPE)：DilatedReparamBlock + BN
        self.lepe = nn.Sequential(
            DilatedReparamBlock(
                dim, kernel_size=kernel_size,
                deploy=deploy,
                use_sync_bn=False,
                attempt_use_lk_impl=use_gemm
            ),
            nn.BatchNorm2d(dim),
        )

        # query: 来自 x 的一半通道
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        # key: 来自 ctx 分支 (这里直接用 x 的另一半通道 + 全局池)
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        # 生成动态权重：49 -> k^2 + smk^2
        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.get_rpb()

        if not _HAS_NATTEN:
            print("[ContMix] NATTEN not available, using slow PyTorch na2d_av() fallback.")

    # ---------------- RPB 相关 ----------------

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(
            torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1)
        )
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(
            torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2)
        )
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size: int):
        """
        生成 offset index，用于从 RPB 张量中索引出对应位置的偏置。
        """
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return idx_h, idx_w, idx_k

    def apply_rpb(self,
                  attn: torch.Tensor,
                  rpb: torch.Tensor,
                  height: int,
                  width: int,
                  kernel_size: int,
                  idx_h: torch.Tensor,
                  idx_w: torch.Tensor,
                  idx_k: torch.Tensor):
        """
        把相对位置偏置加到 attn 上。
        attn 形状：(B, heads, H, W, K^2)
        """
        # ------- 关键修复：保证所有索引和 rpb 与 attn 在同一 device，避免 CPU/CUDA 混用 -------
        device = attn.device

        rpb = rpb.to(device)
        idx_h = idx_h.to(device=device, dtype=torch.long)
        idx_w = idx_w.to(device=device, dtype=torch.long)
        idx_k = idx_k.to(device=device, dtype=torch.long)

        # num_repeat_* 直接在当前 device 上创建
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long, device=device)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long, device=device)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)

        # 计算每个 (h,w) 位置对应的 rpb 索引
        bias_hw = (
            idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2 * kernel_size - 1)
        ) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k                     # (H*?, ?, K2)
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))       # (H*W, K2)
        bias_idx = torch.flip(bias_idx, [0])

        # rpb: (heads, R, R) -> (heads, H*W, K2)
        rpb_flat = torch.flatten(rpb, 1, 2)[:, bias_idx]             # (heads, H*W, K2)
        rpb_flat = rpb_flat.reshape(
            1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2)
        )
        return attn + rpb_flat

    # ---------------- forward ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, dim, H, W)
        """
        B, C, H, W = x.shape

        # LEPE 分支
        lepe = self.lepe(x)  # (B,C,H,W)

        # 通道一分为二：一半用于 query，一半用于 key/value
        q_src, k_src = torch.chunk(x, 2, dim=1)  # (B,C/2,H,W)

        # query
        query = self.weight_query(q_src) * self.scale    # (B,C/2,H,W)
        # key
        key = self.weight_key(k_src)                     # (B,C/2,7,7) --> 在 weight_proj 里用

        # 拆多头
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key,   'b (g c) h w -> b g c (h w)', g=self.num_heads)

        # 注意力 logits
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')     # (B,g,N,49)
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()  # (B,49,g,N)

        # 生成动态核权重：(B, k^2+smk^2, g, N)
        weight = self.weight_proj(weight)                              # 1x1 conv
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        # 拆成小核/大核两套注意力
        attn1, attn2 = torch.split(
            weight,
            split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2],
            dim=-1
        )

        # 加相对位置偏置
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)

        # softmax
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)

        # value: 原始 x 拆多头
        value = rearrange(
            x, 'b (m g c) h w -> m b g h w c',
            m=2, g=self.num_heads
        )  # value[0], value[1] 分给两个核

        # 局部注意加权聚合（na2d_av）
        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)   # (B,g,H,W,C_head)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        # concat heads & 通道
        x_out = torch.cat([x1, x2], dim=1)                         # (B,2g,H,W,C_head)
        x_out = rearrange(x_out, 'b g h w c -> b (g c) h w', h=H, w=W)

        # 1x1 投影 + 加 LEPE
        x_out = self.dyconv_proj(x_out)
        x_out = x_out + lepe

        return x_out


if __name__ == "__main__":
    # 简单自测
    inp = torch.randn(1, 64, 32, 32)
    model = ContMix(dim=64, ctx_dim=32, kernel_size=7, smk_size=5, num_heads=2)
    out = model(inp)
    print("输入张量形状:", inp.shape)
    print("输出张量形状:", out.shape)
