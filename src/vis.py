import argparse
import numpy as np
import open3d as o3d
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
CUR = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(CUR)))

from configs import eval_config_params
from data import ModelNet40
from models import ROPNet
from utils import npy2pcd, pcd2npy, vis_pcds


def vis_ROPNet(args, test_loader):
    model = ROPNet(args)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        for i, (tgt_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):

            if args.cuda:
                tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()

            B, N, _ = src_cloud.size()
            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=2)
            pred_src = results['pred_src']

            tgt_cloud = torch.squeeze(tgt_cloud[..., :3]).cpu().numpy()
            src_cloud = torch.squeeze(src_cloud[..., :3]).cpu().numpy()
            pred_cloud = torch.squeeze(pred_src[-1]).cpu().numpy()
            tgt_cloud = npy2pcd(tgt_cloud)
            src_cloud = npy2pcd(src_cloud)
            tgt_cloud.paint_uniform_color([1, 0, 0])
            src_cloud.paint_uniform_color([0, 1, 0])
            pred_cloud = npy2pcd(pred_cloud)
            pred_cloud.paint_uniform_color([0, 0, 1])
            vis_pcds([tgt_cloud, src_cloud, pred_cloud], need_color=False)


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

    vis_ROPNet(args, test_loader)
