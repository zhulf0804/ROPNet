import numpy as np
import torch
from metrics import Error_R, Error_t, anisotropic_R_error, anisotropic_t_error
from utils import inv_R_t


def compute_metrics(R, t, gtR, gtt):
    inv_R, inv_t = inv_R_t(gtR, gtt)
    if isinstance(R, torch.Tensor):
        R = R.cpu().detach().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    if isinstance(inv_R, torch.Tensor):
        inv_R = inv_R.cpu().detach().numpy()
    if isinstance(inv_t, torch.Tensor):
        inv_t = inv_t.cpu().detach().numpy()

    cur_r_mse, cur_r_mae = anisotropic_R_error(R, inv_R)
    cur_t_mse, cur_t_mae = anisotropic_t_error(t, inv_t)
    cur_r_isotropic = Error_R(R, inv_R)
    cur_t_isotropic = Error_t(t, inv_t, inv_R)
    return cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
           cur_t_isotropic


def summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic):
    r_mse = np.concatenate(r_mse, axis=0)
    r_mae = np.concatenate(r_mae, axis=0)
    t_mse = np.concatenate(t_mse, axis=0)
    t_mae = np.concatenate(t_mae, axis=0)
    r_isotropic = np.concatenate(r_isotropic, axis=0)
    t_isotropic = np.concatenate(t_isotropic, axis=0)

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        np.sqrt(np.mean(r_mse) + 1e-8), np.mean(r_mae), \
        np.sqrt(np.mean(t_mse) + 1e-8), np.mean(t_mae), np.mean(r_isotropic), \
        np.mean(t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def print_metrics(method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic):
    print('='*20, method, '='*20)
    print('time: {:.4f} s, mean: {:.4f} s'.format(np.sum(dura), np.mean(dura)))
    print('Error R error: {:.4f}'.format(r_isotropic))
    print('Error t error: {:.4f}'.format(t_isotropic))
    print('anisotropic mse R error: {:.4f}'.format(r_mse))
    print('anisotropic mae R error: {:.4f}'.format(r_mae))
    print('anisotropic mse t error : {:.4f}'.format(t_mse))
    print('anisotropic mae t error: {:.4f}'.format(t_mae))


def print_train_info(results):
    print('Loss: {:.4f}, Error R: {:.4f}, Error t: {:.4f}, '
          'anisotropic R(mse, mae): {:.4f}, {:.4f}, '
          'anisotropic t(mse, mae): {:.4f}, {:.4f}'.
          format(results['loss'], results['r_isotropic'],
                 results['t_isotropic'], results['r_mse'], results['r_mae'],
                 results['t_mse'], results['t_mae']))
