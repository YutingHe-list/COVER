import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.UNet import UNet_base
from utils.STN import SpatialTransformer


class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, dimensions='2D', norm=nn.GroupNorm):
        super().__init__()
        self.norm = norm(1, dim)
        if dimensions == '2D':
            self.proj = nn.Conv2d(in_channels, dim, 1)
        elif dimensions == '3D':
            self.proj = nn.Conv3d(in_channels, dim, 1)
        else:
            assert "dimensions should be 2D or 3D"

    def forward(self, feat):
        feat = self.norm(self.proj(feat))
        return feat

class MoV2D(nn.Module):
    def __init__(self, amp):
        super().__init__()
        self.amp = amp
        self.kernel_size = amp * 2 + 1
        self.softmax = nn.Softmax(dim=1)

        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [amp * 2 + 1] * 2]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def makeV(self, num_heads):
        v = self.grid.reshape((self.kernel_size)**2, 2)[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
        v = v.repeat(1, 1, 1, num_heads, 1, 1)
        return v

    def forward_self(self, q, k, num_heads):
        """ Forward function.
        Args:
            q: query features with shape of (B, C, H, W)
            k: key features with shape of (B, C, H, W)
        """

        B, C, H, W = q.shape
        q = q.reshape(B, C // num_heads, num_heads, H, W)  # (B, C//num_heads, num_heads, H, W)
        k = k.reshape(B, C // num_heads, num_heads, H, W)  # (B, C//num_heads, num_heads, H, W)

        head_dim = C // num_heads
        scale = head_dim ** -0.5

        q = q * scale
        dim = (self.amp, self.amp, self.amp, self.amp)
        k = F.pad(k, dim, "constant")  # (B, C//num_heads, num_heads, H+2amp, W+2amp)
        mask_k = torch.ones(size=(B, 1, num_heads, H, W), device=k.device)
        mask_k = F.pad(mask_k, dim, "constant")

        attn = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                mask_k_idx = torch.where(mask_k[:, :, :, i:i + H, j:j + W] < 0.5)
                attn_i = torch.sum(q * k[:, :, :, i:i + H, j:j + W], dim=1, keepdim=True)
                attn_i[mask_k_idx] = -1000.
                attn.append(attn_i) # (B, 1, num_heads, H, W)

        attn = torch.cat(attn, dim=1)   # (B, kernel_size**3, num_head, H, W)

        attn = self.softmax(attn)[:, :, np.newaxis, :, :, :] # (B, kernel_size**3, 1, num_head, H, W)

        v = self.makeV(num_heads)  # 1, kernel_size**3, 3, num_heads, 1, 1

        v_ = torch.sum(attn * v, dim=1)  # (B, 3, num_heads, H, W)

        x_v = torch.mean(v_, dim=2)

        return x_v

    def forward(self, x_q, x_k, num_heads):
        x = self.forward_self(x_q, x_k, num_heads)
        return x


class MoV3D(nn.Module):
    def __init__(self, amp):
        super().__init__()
        self.amp = amp
        self.kernel_size = amp * 2 + 1
        self.softmax = nn.Softmax(dim=1)

        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [amp * 2 + 1] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def makeV(self, num_heads):
        v = self.grid.reshape((self.kernel_size)**3, 3)[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        v = v.repeat(1, 1, 1, num_heads, 1, 1, 1)
        return v

    def forward_self(self, q, k, num_heads):
        """ Forward function.
        Args:
            q: query features with shape of (B, C, H, W, D)
            k: key features with shape of (B, C, H, W, D)
        """

        B, C, H, W, D = q.shape
        q = q.reshape(B, C // num_heads, num_heads, H, W, D)  # (B, C//num_heads, num_heads, H, W, D)
        k = k.reshape(B, C // num_heads, num_heads, H, W, D)  # (B, C//num_heads, num_heads, H, W, D)

        head_dim = C // num_heads
        scale = head_dim ** -0.5

        q = q * scale
        dim = (self.amp, self.amp, self.amp, self.amp, self.amp, self.amp)
        k = F.pad(k, dim, "constant")  # (B, C//num_heads, num_heads, H+2amp, W+2amp, D+2amp)
        mask_k = torch.ones(size=(B, 1, num_heads, H, W, D), device=k.device)
        mask_k = F.pad(mask_k, dim, "constant")

        attn = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for s in range(self.kernel_size):
                    mask_k_idx = torch.where(mask_k[:, :, :, i:i + H, j:j + W, s:s + D] < 0.5)
                    attn_i = torch.sum(q * k[:, :, :, i:i + H, j:j + W, s:s + D], dim=1, keepdim=True)
                    attn_i[mask_k_idx] = -1000.
                    attn.append(attn_i) # (B, 1, num_heads, H, W, D)

        attn = torch.cat(attn, dim=1)   # (B, kernel_size**3, num_head, H, W, D)

        attn = self.softmax(attn)[:, :, np.newaxis, :, :, :, :] # (B, kernel_size**3, 1, num_head, H, W, D)

        v = self.makeV(num_heads)  # 1, kernel_size**3, 3, num_heads, 1, 1, 1

        v_ = torch.sum(attn * v, dim=1)  # (B, 3, num_heads, H, W, D)

        x_v = torch.mean(v_, dim=2)

        return x_v

    def forward(self, x_q, x_k, num_heads):
        x = self.forward_self(x_q, x_k, num_heads)
        return x

class COVER(nn.Module):
    def __init__(self,
                 backbone=UNet_base,
                 dimensions='2D',
                 n_channels=1,
                 chan=(32, 64, 128, 256, 512, 256, 128, 64, 32),
                 head_dim=8,
                 num_heads=[4, 4, 4, 1, 1],
                 amp=2):
        super(COVER, self).__init__()

        self.backbone = backbone(n_channels=n_channels, chs=chan, dimensions=dimensions)

        if dimensions == '2D':
            self.mov = MoV2D(amp=amp)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif dimensions == '3D':
            self.mov = MoV3D(amp=amp)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            assert "dimensions should be 2D or 3D"
        self.num_heads = num_heads

        depth = (len(chan) + 1) // 2
        self.proj = nn.ModuleList()
        for i in range(depth):
            self.proj.append(ProjectionLayer(chan[len(chan)//2+i], head_dim * num_heads[i], dimensions=dimensions))

        self.stn = SpatialTransformer()

    def VPA(self, M, F):

        q, k = self.proj[0](F[-1]), self.proj[0](M[-1])
        w = self.mov(q, k, self.num_heads[0]) * 2
        w = self.upsample(w)
        flow = w

        for i in range(1, len(self.proj)-1):
            M[-i-1] = self.stn(M[-i-1], flow)
            q, k = self.proj[i](F[-i-1]), self.proj[i](M[-i-1])
            w = self.mov(q, k, self.num_heads[i]) * 2
            w = self.upsample(w)
            flow = self.stn(self.upsample(2 * flow), w) + w

        M[0] = self.stn(M[0], flow)
        q, k = self.proj[-1](F[0]), self.proj[-1](M[0])
        w = self.mov(q, k, self.num_heads[-1])
        flow = self.stn(flow, w) + w

        return flow

    def forward(self, M, F):
        M1, M2, M3, M4, M5 = self.backbone(M)
        F1, F2, F3, F4, F5 = self.backbone(F)
        flow = self.VPA(M=[M1, M2, M3, M4, M5], F=[F1, F2, F3, F4, F5])

        return flow, M1, F1