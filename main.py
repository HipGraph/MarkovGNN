import os.path as osp
import argparse
import sys
import numpy as np
from tqdm import tqdm
import torch, random
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, MGCNConv   # noqa
import networkx as nx
from sklearn.cluster import KMeans
import community as comm
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, dense_to_sparse
import threading
from torch_sparse import spspmm
from torch_geometric.data import Data
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from utils import readInput3f, readInput2f, loadPyGDataset
from markov import markov_process_agg, markov_process_disj
from models import MarkovGCNR, GCN
from train import train

# disable this line if do not want to fix random seed
if True:
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

def helper(data, args):
    eps = args.eps
    inflate = args.inflate
    normrow = args.normrow
    nlayers = args.nlayers
    ndim = args.ndim
    alpha = args.alpha
    droprate = args.droprate
    nepoch = args.epoch
    lrate = args.lrate
    useleakyrelu = args.useleakyrelu
    # define a model
    if args.use_gcn:
        model = GCN(ndim, nlayers, len(set(data.y.tolist())), data.x, data.edge_index, data.edge_attr, droprate, useleakyrelu==1, alpha)
    else:
        if args.markov_agg:
            (edge_index, edge_weight) = markov_process_agg(data, eps, inflate, nlayers, normrow == 1, args.keepmax == 1, args.debug == 1)
        else:
            (edge_index, edge_weight) = markov_process_disj(data, eps, inflate, nlayers, normrow == 1, args.keepmax == 1, args.debug == 1)
        if False:
            print("layer-wise edge shape", edge_index)
        model = MarkovGCNR(ndim, nlayers, len(set(data.y.tolist())), data.x, edge_index, edge_weight, droprate, useleakyrelu==1, alpha)
    #define an optimizer
    optimizerdict = []
    for l in range(nlayers-1):
        optimizerdict.append(dict(params=model.convs[l].parameters(), weight_decay=5e-4))
    optimizerdict.append(dict(params=model.convs[nlayers-1].parameters(), weight_decay=0))
    optimizer = torch.optim.Adam(optimizerdict, lr=lrate)
    train(model, data, optimizer, nepoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--edgelist', required = False, type = str, help = 'Input edgelist file for the graph.')
    parser.add_argument('--label', required = False, type = str, help = 'Input file for the true labels.')
    parser.add_argument('--feature', required = False, type = str, help = 'Feature file.')
    parser.add_argument('--dataset', default = 'Cora', required = False, type = str, help = 'PyG dataset name.')
    parser.add_argument('--use_gcn', action='store_true', help='Use vanilla GCN model.')
    parser.add_argument('--markov_agg', action='store_true', help='Use markov agg version.')
    parser.add_argument('--eps', default = 0.25, required=False, type=float, help='Use threshold.')
    parser.add_argument('--normrow', default = 1, required = False, type = int, help='Normalization row/column.')
    parser.add_argument('--keepmax', default = 1, required=False, type=int, help='Take max entries based on eps.')
    parser.add_argument('--alpha', default = 0.5, required=False, type=float, help='Value of alpha.')
    parser.add_argument('--inflate', default = 1.5, required=False, type=float, help='Inflattion parameter.')
    parser.add_argument('--oneindexed', default = 0, required = False, type = int, help = 'Node index type in file 0/1.')
    parser.add_argument('--onelabeled', default = 0, required = False, type = int, help='Label starting ids.')
    parser.add_argument('--nlayers', default = 2, required=False, type=int, help='Number of hidden layers in the GNN.')
    parser.add_argument('--ndim', default = 64, required=False, type=int, help='Number of hidden units in the GNN.')
    parser.add_argument('--useleakyrelu', default = 0, required=False, type=int, help='Use leakyrelu activation.')
    parser.add_argument('--lrate', default = 0.01, required = False, type = float, help = 'Learning Rate')
    parser.add_argument('--droprate', default = 0.5, required = False, type = float, help = 'Dropout  Rate')
    parser.add_argument('--epoch', default = 100, required=False, type=int, help='Number of epoch.')
    parser.add_argument('--debug', default = 1, required=False, type=int, help='Disable debug mode.')

    args = parser.parse_args()
    edgelistf = args.edgelist
    labelf = args.label
    dataset_name = args.dataset
    featuref = args.feature
    oneindexed = args.oneindexed
    onelabeled = args.onelabeled

    if edgelistf and labelf and featuref:
        data = readInput3f(edgelistf, labelf, featuref, oneindexed == 1, onelabeled == 1, args.debug == 1)
    elif edgelistf and labelf:
        data = readInput2f(edgelistf, labelf, oneindexed == 1, onelabeled == 1, args.debug == 1)
    else:
        data = loadPyGDataset(dataset_name)
    print(data)
    helper(data, args)
