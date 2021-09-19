import torch
import torch.nn as nn
from utils import batch_transform, square_dists


def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e10
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(square_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]


def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)
    grouped_inds[dists > radius ** 2] = N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    if M < 0:
        new_xyz = xyz
    else:
        new_xyz = gather_points(xyz, fps(xyz, M))
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


def weighted_icp(src, tgt, weights, _EPS = 1e-8):
    """Compute rigid transforms between two point sets

    Args:
        src (torch.Tensor): (B, M, 3) points
        tgt (torch.Tensor): (B, M, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        R, t, transformed_src: (B, 3, 3), (B, 3), (B, M, 3)

    Modified from open source code:
        https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py
    """
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_src = torch.sum(src * weights_normalized, dim=1)
    centroid_tgt = torch.sum(tgt * weights_normalized, dim=1)
    src_centered = src - centroid_src[:, None, :]
    tgt_centered = tgt - centroid_tgt[:, None, :]
    cov = src_centered.transpose(-2, -1) @ (tgt_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_src[:, :, None] + centroid_tgt[:, :, None]
    translation = torch.squeeze(translation, -1)
    transformed_src = batch_transform(src, rot_mat, translation)
    return rot_mat, translation, transformed_src