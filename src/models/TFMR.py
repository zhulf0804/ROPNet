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

from utils import batch_transform, angle
from models import gather_points, sample_and_group, weighted_icp


class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'gn_{i}',
                                   nn.GroupNorm(out_dim // 32, out_dim))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''

        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x


class LocalFeatue(nn.Module):
    def __init__(self, radius, K, in_dim, out_dims):
        super(LocalFeatue, self).__init__()
        self.radius = radius
        self.K = K
        self.local_feature_fused = LocalFeatureFused(in_dim=in_dim,
                                                     out_dims=out_dims)

    def forward(self, feature, xyz, permute=False, use_ppf=False):
        '''
        :param feature: (B, N, C1) or (B, C1, N) for permute
        :param xyz: (B, N, 3)
        :return: (B, C2, N)
        '''
        if permute:
            feature = feature.permute(0, 2, 1).contiguous()
        new_xyz, new_points, grouped_inds, grouped_xyz = \
            sample_and_group(xyz=xyz,
                             points=feature,
                             M=-1,
                             radius=self.radius,
                             K=self.K)
        if use_ppf:
            nr_d = angle(feature[:, :, None, :], grouped_xyz)
            ni_d = angle(new_points[..., 3:], grouped_xyz)
            nr_ni = angle(feature[:, :, None, :], new_points[..., 3:])
            d_norm = torch.norm(grouped_xyz, dim=-1)
            ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1) # (B, N, K, 4)
            new_points = torch.cat([new_points[..., :3], ppf_feat], dim=-1)
        xyz = torch.unsqueeze(xyz, dim=2).repeat(1, 1, self.K, 1)
        new_points = torch.cat([xyz, new_points], dim=-1)
        feature_local = new_points.permute(0, 3, 2, 1).contiguous() # (B, C1 + 3, K, M)
        feature_local = self.local_feature_fused(feature_local)
        return feature_local


class OverlapAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(OverlapAttentionBlock, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.GroupNorm(channels // 32, channels)
        self.act = nn.ReLU()

    def forward(self, x, ol_score):
        '''
        :param x: (B, C, N)
        :param ol: (B, N)
        :return: (B, C, N)
        '''
        B, C, N = x.size()
        x_q = self.q_conv(x).permute(0, 2, 1).contiguous() # B, N, C
        x_k = self.k_conv(x) # B, C, N
        x_v = self.v_conv(x)
        attention = torch.bmm(x_q, x_k) # B, N, N
        if ol_score is not None:
            ol_score = torch.unsqueeze(ol_score, dim=-1).repeat(1, 1, N) # (B, N, N)
            attention = ol_score * attention
        attention = torch.softmax(attention, dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # B, C, N
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class OverlapAttention(nn.Module):
    def __init__(self, dim):
        super(OverlapAttention, self).__init__()
        self.overlap_attention1 = OverlapAttentionBlock(dim)
        self.overlap_attention2 = OverlapAttentionBlock(dim)
        self.overlap_attention3 = OverlapAttentionBlock(dim)
        self.overlap_attention4 = OverlapAttentionBlock(dim)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim*4, dim*4, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim*4),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, ol):
        x1 = self.overlap_attention1(x, ol)
        x2 = self.overlap_attention2(x1, ol)
        x3 = self.overlap_attention3(x2, ol)
        x4 = self.overlap_attention4(x3, ol)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        return x


class TFMRModule(nn.Module):
    def __init__(self, args):
        super(TFMRModule, self).__init__()
        self.N1s = [args.train_N1, args.test_N1]
        self.M1s = [args.train_M1, args.test_M1]
        self.top_probs = [args.train_top_prob, args.test_top_prob]
        self.similarity_topks = [args.train_similarity_topk, args.test_similarity_topk]
        self.use_ppf = args.use_ppf
        in_dim = 10 if self.use_ppf else 6
        self.local_features = LocalFeatue(radius=args.radius,
                                          K=args.num_neighbors,
                                          in_dim=in_dim,
                                          out_dims=[128, 256, args.feat_dim])
        self.overlap_attention = OverlapAttention(args.feat_dim)

        # to reuse features for tgt
        self.tgt_info = {'f_y_atten': None,
                         'tgt': None}

    def forward(self, src, tgt, x_ol_score, y_ol_score, train, iter,
                normal_src=None, normal_tgt=None):
        '''

        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :param x_ol_score: (B, N)
        :param y_ol_score: (B, M)
        :param train: bool, for hyperparameters selection
        :param iter: int
        :param normal_src: (B, N, 3)
        :param normal_tgt: (B, M, 3)
        :return: src: (B, N2, 3); tgt_corr: (B, N2, 3); icp_weights: (B, N2);
        similarity_max_inds: (B, N2), for overlap evaluation.
        '''
        '''

        :param src: 
        :param tgt: 
        :param x_ol_score: 
        return: correspondences points: (N2, 3), (N2, 3)
        '''
        B, _, _ = src.size()
        if train:
            N1, M1, top_prob = self.N1s[0], self.M1s[0], self.top_probs[0]
            similarity_topk = self.similarity_topks[0]
        else:
            N1, M1, top_prob = self.N1s[1], self.M1s[1], self.top_probs[1]
            similarity_topk = self.similarity_topks[1]

        # point feature extraction for tgt
        if self.use_ppf and normal_src is not None:
            f_x_p = self.local_features(feature=normal_src,
                                        xyz=src,
                                        use_ppf=True)
        else:
            f_x_p = self.local_features(feature=None,
                                        xyz=src)

        f_x_attn = self.overlap_attention(f_x_p, ol=None). \
            permute(0, 2, 1).contiguous()
        f_x_atten = f_x_attn / (torch.norm(f_x_attn, dim=-1,
                                        keepdim=True) + 1e-8)  # (B, N, C)
        x_ol_score, x_ol_inds = torch.sort(x_ol_score, dim=-1, descending=True)
        x_ol_inds = x_ol_inds[:, :N1]
        f_x_atten = gather_points(f_x_atten, x_ol_inds)  # (B, N1, C)
        src = gather_points(src, x_ol_inds)

        if iter == 0:
            # point feature extraction for tgt
            if self.use_ppf and normal_tgt is not None:
                f_y_p = self.local_features(feature=normal_tgt,
                                            xyz=tgt,
                                            use_ppf=True)
            else:
                f_y_p = self.local_features(feature=None,
                                            xyz=tgt)

            f_y_atten = self.overlap_attention(f_y_p, ol=None). \
                permute(0, 2, 1).contiguous()
            f_y_atten = f_y_atten / (torch.norm(f_y_atten, dim=-1,
                                                keepdim=True) + 1e-8)  # (B, M, C)
            y_ol_score, y_ol_inds = torch.sort(y_ol_score, dim=-1, descending=True)
            y_ol_inds = y_ol_inds[:, :M1]
            f_y_atten = gather_points(f_y_atten, y_ol_inds)  # (B, M1, C)
            tgt = gather_points(tgt, y_ol_inds)

            self.tgt_info = {'f_y_atten': f_y_atten,
                             'tgt': tgt}
        else:
            f_y_atten, tgt = self.tgt_info['f_y_atten'], self.tgt_info['tgt']

        similarity = torch.bmm(f_x_atten, f_y_atten.permute(0, 2, 1).contiguous()) # (B, N1, M1)

        # feature matching removal
        N2 = int(top_prob * N1)  # train
        similarity_max = torch.max(similarity, dim=-1)[0]  # (B, N1)
        similarity_max_inds = \
            torch.sort(similarity_max, dim=-1, descending=True)[1][:, :N2]
        src = gather_points(src, similarity_max_inds)

        # generate correspondences
        similarity = gather_points(similarity, similarity_max_inds) # (B, N2, M1)
        x_ol_score = torch.squeeze(
            gather_points(torch.unsqueeze(x_ol_score, dim=-1),
                          similarity_max_inds), dim=-1)
        # find topk points in feature space
        device = similarity.device
        similarity_topk_inds = \
            torch.topk(similarity, k=similarity_topk, dim=-1)[1]  # (B, N2, topk)
        mask = torch.zeros_like(similarity).to(device).detach()
        inds1 = torch.arange(B, dtype=torch.long).to(device). \
            reshape((B, 1, 1)).repeat((1, N2, similarity_topk))
        inds2 = torch.arange(N2, dtype=torch.long).to(device). \
            reshape((1, N2, 1)).repeat((B, 1, similarity_topk))
        mask[inds1, inds2, similarity_topk_inds] = 1
        similarity = similarity * mask

        weights = similarity / \
                  (torch.sum(similarity, dim=-1, keepdim=True) + 1e-8)
        tgt_corr = torch.bmm(weights, tgt)
        icp_weights = x_ol_score[:, :N2]

        return src, tgt_corr, icp_weights, similarity_max_inds
