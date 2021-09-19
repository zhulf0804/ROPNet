import math
import numpy as np
from scipy.spatial.transform import Rotation


def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def Error_t(t1, t2, r2):
    '''
    Calculate isotropic translation error between t1 and t2.
    :param t1: shape=(B, 3), pred_t
    :param t2: shape=(B, 3), gtt
    :param R2: shape=(B, 3, 3), gtR
    :return:
    '''
    r2 = r2.transpose(0, 2, 1)
    t2 = np.squeeze(- r2 @ t2[..., None], axis=-1)
    error_t = np.squeeze(r2 @ t1[..., None], axis=-1) + t2
    error_t = np.linalg.norm(error_t, axis=-1)
    return error_t


def anisotropic_R_error(r1, r2, seq='xyz', degrees=True):
    '''
    Calculate mse, mae euler agnle error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2)**2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)
    return r_mse, r_mae


def anisotropic_t_error(t1, t2):
    '''
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return t_mse, t_mae
