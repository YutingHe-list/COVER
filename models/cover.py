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

class MoV(nn.Module):
    def __init__(self, input_dim, amp, num_heads):
        super().__init__()
        self.amp = amp
        self.kernel_size = amp * 2 + 1
        self.input_dim = input_dim
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5

        self.softmax = nn.Softmax(dim=1)

        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [amp * 2 + 1] * 2]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def makeV(self):
        v = self.grid.reshape((self.kernel_size)**2, 2)[np.newaxis, :, :, np.newaxis, np.newaxis, np.newaxis]
        v = v.repeat(1, 1, 1, self.num_heads, 1, 1)
        return v

    def forward_self(self, q, k):
        """ Forward function.
        Args:
            q: query features with shape of (B, C, H, W)
            k: key features with shape of (B, C, H, W)
        """

        B, C, H, W = q.shape
        q = q.reshape(B, C // self.num_heads, self.num_heads, H, W)  # (B, C//num_heads, num_heads, H, W)
        k = k.reshape(B, C // self.num_heads, self.num_heads, H, W)  # (B, C//num_heads, num_heads, H, W)

        q = q * self.scale
        dim = (self.amp, self.amp, self.amp, self.amp)
        k = F.pad(k, dim, "constant")  # (B, C//num_heads, num_heads, H+2amp, W+2amp)
        mask_k = torch.ones(size=(B, 1, self.num_heads, H, W), device=k.device)
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

        v = self.makeV()  # 1, kernel_size**3, 3, num_heads, 1, 1

        v_ = torch.sum(attn * v, dim=1)  # (B, 3, num_heads, H, W)

        x_v = torch.mean(v_, dim=2)

        return x_v

    def forward(self, x_q, x_k):
        x = self.forward_self(x_q, x_k)
        return x

class COVER(nn.Module):
    def __init__(self, backbone=UNet_base, dimensions='2D', n_channels=1,
                 chan=(32, 64, 128, 256, 512, 256, 128, 64, 32),
                 head_dim=8,
                 num_heads=[4, 4, 4, 1, 1],
                 amp=2):
        super(COVER, self).__init__()

        self.backbone = backbone(n_channels=n_channels, chs=chan, dimensions=dimensions)

        self.upsample_bilin = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.projblock1 = ProjectionLayer(chan[-1], dim=head_dim * num_heads[-1], dimensions=dimensions)
        self.mov1 = MoV(input_dim=head_dim * num_heads[-1], num_heads=num_heads[-1], amp=amp)

        self.projblock2 = ProjectionLayer(chan[-2], dim=head_dim * num_heads[-2], dimensions=dimensions)
        self.mov2 = MoV(input_dim=head_dim * num_heads[-2], num_heads=num_heads[-2], amp=amp)

        self.projblock3 = ProjectionLayer(chan[-3], dim=head_dim * num_heads[-3], dimensions=dimensions)
        self.mov3 = MoV(input_dim=head_dim * num_heads[-3], num_heads=num_heads[-3], amp=amp)

        self.projblock4 = ProjectionLayer(chan[-4], dim=head_dim * num_heads[-4], dimensions=dimensions)
        self.mov4 = MoV(input_dim=head_dim * num_heads[-4], num_heads=num_heads[-4], amp=amp)

        self.projblock5 = ProjectionLayer(chan[-5], dim=head_dim * num_heads[-5], dimensions=dimensions)
        self.mov5 = MoV(input_dim=head_dim * num_heads[-5], num_heads=num_heads[-5], amp=amp)

        self.stn = SpatialTransformer()

    def VPA(self, M, F):
        M1, M2, M3, M4, M5 = M
        F1, F2, F3, F4, F5 = F

        q5, k5 = self.projblock5(F5), self.projblock5(M5)
        w = self.mov5(q5, k5)*2
        w = self.upsample_bilin(w)
        flow = w

        M4 = self.stn(M4, flow)
        q4, k4 = self.projblock4(F4), self.projblock4(M4)
        w = self.mov4(q4, k4)*2
        w = self.upsample_bilin(w)
        flow = self.stn(self.upsample_bilin(2 * flow), w) + w

        M3 = self.stn(M3, flow)
        q3, k3 = self.projblock3(F3), self.projblock3(M3)
        w = self.mov3(q3, k3)*2
        w = self.upsample_bilin(w)
        flow = self.stn(self.upsample_bilin(2 * flow), w) + w

        M2 = self.stn(M2, flow)
        q2, k2 = self.projblock2(F2), self.projblock2(M2)
        w = self.mov2(q2, k2)*2
        w = self.upsample_bilin(w)
        flow = self.stn(self.upsample_bilin(2 * flow), w) + w

        M1 = self.stn(M1, flow)
        q1, k1 = self.projblock1(F1), self.projblock1(M1)
        w = self.mov1(q1, k1)
        flow = self.stn(flow, w) + w

        return flow

    def forward(self, M, F):
        M1, M2, M3, M4, M5 = self.backbone(M)
        F1, F2, F3, F4, F5 = self.backbone(F)
        flow = self.VPA((M1, M2, M3, M4, M5), (F1, F2, F3, F4, F5))

        return flow, M1, F1