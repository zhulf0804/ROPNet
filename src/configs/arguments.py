import argparse


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters',
                                     add_help=False)
    ## dataset
    parser.add_argument('--root', required=True, help='the data path')
    parser.add_argument('--npts', type=int, default=1024,
                        help='the points number of each pc for training')
    parser.add_argument('--unseen', action='store_true',
                        help='whether to use unseen mode')
    parser.add_argument('--p_keep', type=list, default=[0.7, 0.7],
                        help='the keep ratio for partial registration')
    parser.add_argument('--ao', action='store_true',
                        help='whether to use asymmetric objects')
    parser.add_argument('--normal', default=True,
                        help='whether to use normal data')
    parser.add_argument('--noise', action='store_true',
                        help='whether to add noise when test')
    parser.add_argument('--use_ppf', default=True,
                        help='whether to use_ppf as input feature')
    ## model
    parser.add_argument('--train_N1', type=int, default=448, help='')
    parser.add_argument('--train_M1', type=int, default=717, help='')
    parser.add_argument('--train_similarity_topk', type=int, default=3, help='')
    parser.add_argument('--test_N1', type=int, default=448, help='')
    parser.add_argument('--test_M1', type=int, default=717, help='')
    parser.add_argument('--test_similarity_topk', type=int, default=1,
                        help='')
    parser.add_argument('--train_top_prob', type=float, default=0.6,
                        help='')
    parser.add_argument('--test_top_prob', type=float, default=0.4,
                        help='')
    # logs
    parser.add_argument('--resume', default='',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--log_freq', type=int, default=8,
                        help='the frequency[steps] to save the summary')
    parser.add_argument('--eval_freq', type=int, default=4,
                        help='the frequency[steps] to eval the val set')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='the frequency[epoch] to save the checkpoints')
    return parser


def train_config_params():
    parser = argparse.ArgumentParser(parents=[config_params()])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoches', type=int, default=600)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N',
                        help='Max num of neighbors to use')
    parser.add_argument('--feat_dim', type=int, default=192,
                        help='Feature dimension (to compute distances on). '
                             'Other numbers will be scaled accordingly')
    args = parser.parse_args()
    return args


def eval_config_params():
    parser = argparse.ArgumentParser(parents=[config_params()])
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N',
                        help='Max num of neighbors to use')
    parser.add_argument('--feat_dim', type=int, default=192,
                        help='Feature dimension (to compute distances on)')
    parser.add_argument('--checkpoint', default='',
                        help='the path to the trained checkpoint')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use the cuda')
    parser.add_argument('--show', action='store_true',
                        help='whether to visualize')
    args = parser.parse_args()
    return args
