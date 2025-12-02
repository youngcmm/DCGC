import argparse

parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--device', type=str, default="cpu", help='device.')
parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
parser.add_argument('--cluster_num', type=int, default=7, help='cluster number')

# pre-process
parser.add_argument('--n_input', type=int, default=500, help='input feature dimension')
parser.add_argument('--t', type=int, default=2, help="filtering time of Laplacian filters")

# network
parser.add_argument('--beta', type=float, default=1, help='focusing factor beta')
parser.add_argument('--dims', type=int, default=[1500], help='hidden unit')
parser.add_argument('--activate', type=str, default='ident', help='activate function')
parser.add_argument('--tao', type=float, default=0.9, help='high confidence rate')

# training
parser.add_argument('--runs', type=int, default=10, help='runs')
parser.add_argument('--epochs', type=int, default=400, help='training epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--seeds', type=list, default=[162, 290, 272, 301, 282, 107, 13, 189, 278, 191], help='random seeds list')
parser.add_argument('--acc', type=float, default=0, help='acc')
parser.add_argument('--nmi', type=float, default=0, help='nmi')
parser.add_argument('--ari', type=float, default=0, help='ari')
parser.add_argument('--f1', type=float, default=0, help='f1')
parser.add_argument('--lamda', type=float, default=1, help='lamda')

parser.add_argument('--layers_num', type=int, default=1, help='layersNum')
args = parser.parse_args()


# import os
# os.system("python train.py")