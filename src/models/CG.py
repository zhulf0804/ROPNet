import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat


class PointNet(nn.Module):
    def __init__(self, in_dim, gn, out_dims, cls=False):
        super(PointNet, self).__init__()
        self.cls = cls
        l = len(out_dims)
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                         nn.GroupNorm(8, out_dim))
            if self.cls and i != l - 1:
                self.backbone.add_module(f'pointnet_relu_{i}',
                                         nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone(x)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)
        return f, g


class MLPs(nn.Module):
    def __init__(self, in_dim, mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()
        l = len(mlps)
        for i, out_dim in enumerate(mlps):
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if i != l - 1:
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)
        return x


class CGModule(nn.Module):
    def __init__(self, in_dim, gn):
        super(CGModule, self).__init__()
        self.encoder = PointNet(in_dim=in_dim,
                                gn=gn,
                                out_dims=[64, 64, 64, 128, 512])
        self.decoder_ol = PointNet(in_dim=2048,
                                   gn=gn,
                                   out_dims=[512, 512, 256, 2],
                                   cls=True)
        self.decoder_qt = MLPs(in_dim=1024,
                               mlps=[512, 512, 256, 7])

    def forward(self, src, tgt):
        '''
        Context-Guided Model for initial alignment and overlap score.
        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :return: T0: (B, 3, 4), OX: (B, N, 2), OY: (B, M, 2)
        '''
        x = src.permute(0, 2, 1).contiguous()
        y = tgt.permute(0, 2, 1).contiguous()
        f_x, g_x = self.encoder(x)
        f_y, g_y = self.encoder(y)
        concat = torch.cat((g_x, g_y), dim=1)

        # regression initial alignment
        out = self.decoder_qt(concat)
        batch_t, batch_quat = out[:, :3], out[:, 3:] / (
                torch.norm(out[:, 3:], dim=1, keepdim=True) + 1e-8)
        batch_R = batch_quat2mat(batch_quat)
        batch_T = torch.cat([batch_R, batch_t[..., None]], dim=-1)

        # overlap prediction
        g_x_expand = torch.unsqueeze(g_x, dim=-1).expand_as(f_x)
        g_y_expand = torch.unsqueeze(g_y, dim=-1).expand_as(f_y)
        f_x_ensemble = torch.cat([f_x, g_x_expand, g_y_expand,
                                  g_x_expand - g_y_expand], dim=1)
        f_y_ensemble = torch.cat([f_y, g_y_expand, g_x_expand,
                                  g_y_expand - g_x_expand], dim=1)
        x_ol = self.decoder_ol(f_x_ensemble, pooling=False)
        y_ol = self.decoder_ol(f_y_ensemble, pooling=False)

        return batch_T, x_ol, y_ol
