import argparse
import numpy as np
import open3d as o3d
import os
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
CUR = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(CUR)))

from configs import eval_config_params
from data import ModelNet40
from models import ROPNet, gather_points
from utils import npy2pcd, pcd2npy, inv_R_t, batch_transform, square_dists, \
    format_lines, vis_pcds
from metrics import compute_metrics, summary_metrics, print_metrics


def evaluate_ROPNet(args, test_loader):
    model = ROPNet(args)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    src_recalls_op, src_precs_op = [], []
    src_recalls_rop, src_precs_rop = [], []
    with torch.no_grad():
        for i, (tgt_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):

            if args.cuda:
                tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            B, N, _ = src_cloud.size()
            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=2)
            toc = time.time()
            dura.append(toc - tic)
            pred_Ts = results['pred_Ts']
            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]

            # for overlap evaluation
            src_op = results['src_ol1']
            src_rop = results['src_ol2']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            dist_thresh = 0.05
            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                                 inv_t)
            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
            src_ol_gt = torch.min(dists, dim=-1)[0] < dist_thresh * dist_thresh

            gt_transformed_src_op = batch_transform(src_op[..., :3], inv_R,
                                                     inv_t)
            dists_op = square_dists(gt_transformed_src_op, tgt_cloud[..., :3])
            src_op_pred = torch.min(dists_op, dim=-1)[0] < dist_thresh * dist_thresh
            src_prec = torch.sum(src_op_pred, dim=1) / \
                       torch.sum(torch.min(dists_op, dim=-1)[0] > -1)
            src_recall = torch.sum(src_op_pred, dim=1) / torch.sum(src_ol_gt, dim=1)
            src_precs_op.append(src_prec.cpu().numpy())
            src_recalls_op.append(src_recall.cpu().numpy())

            gt_transformed_src_rop = batch_transform(src_rop[..., :3], inv_R,
                                                 inv_t)
            dists_rop = square_dists(gt_transformed_src_rop, tgt_cloud[..., :3])
            src_rop_pred = torch.min(dists_rop, dim=-1)[0] < dist_thresh * dist_thresh
            src_prec = torch.sum(src_rop_pred, dim=1) / \
                       torch.sum(torch.min(dists_rop, dim=-1)[0] > -1)
            src_recall = torch.sum(src_rop_pred, dim=1) / \
                         torch.sum(src_ol_gt, dim=1)
            src_precs_rop.append(src_prec.cpu().numpy())
            src_recalls_rop.append(src_recall.cpu().numpy())

            # src_op = npy2pcd(torch.squeeze(gt_transformed_src_op).cpu().numpy())
            # src_rop = npy2pcd(torch.squeeze(gt_transformed_src_rop).cpu().numpy())
            # tgt_cloud = npy2pcd(torch.squeeze(tgt_cloud[..., :3]).cpu().numpy())
            # vis_pcds([tgt_cloud, src_op])
            # vis_pcds([tgt_cloud, src_rop])

            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)

    print('=' * 20, 'Overlap', '=' * 20)
    print('OP overlap precision: ', np.mean(src_precs_op))
    print('OP overlap recall: ', np.mean(src_recalls_op))
    print('ROP overlap precision: ', np.mean(src_precs_rop))
    print('ROP overlap recall: ', np.mean(src_recalls_rop))

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


if __name__ == '__main__':
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = eval_config_params()
    print(args)

    test_set = ModelNet40(root=args.root,
                          split='test',
                          npts=args.npts,
                          p_keep=args.p_keep,
                          noise=args.noise,
                          unseen=args.unseen,
                          ao=args.ao,
                          normal=args.normal)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        evaluate_ROPNet(args, test_loader)
    print_metrics('ROPNet', dura, r_mse, r_mae, t_mse, t_mae,
                  r_isotropic,
                  t_isotropic)
