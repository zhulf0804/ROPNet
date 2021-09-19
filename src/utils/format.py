import random
import numpy as np
import open3d as o3d
import torch


def readpcd(path, rtype='pcd'):
    assert rtype in ['pcd', 'npy']
    pcd = o3d.io.read_point_cloud(path)
    if rtype == 'pcd':
        return pcd
    npy = np.asarray(pcd.points).astype(np.float32)
    return npy


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    return pcd


def pcd2npy(pcd):
    npy = np.array(pcd.points)
    return npy


def format_lines(points, lines, colors=None):
    '''
    :param points: n x 3
    :param lines:  m x 2
    :param colors: m x 3
    :return:
    '''
    if colors is None:
        colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def vis_pcds(pcds, need_color=True):
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    if need_color:
        for i, pcd in enumerate(pcds):
            color = colors[i] if i < 3 else [random.random() for _ in range(3)]
            pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries(pcds)
