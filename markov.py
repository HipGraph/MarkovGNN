import os.path as osp
import argparse
import sys,time
import numpy as np
from tqdm import tqdm
import torch, random
from torch_sparse import spspmm
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, dense_to_sparse
from torch_scatter import scatter_add

def markov_normalization(edge_index, edge_weight, num_nodes, ntype = 'col'):
    if ntype == 'col':
        _, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = edge_weight * deg_inv[col]
    elif ntype == 'row':
        row, _ = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = edge_weight * deg_inv[row]

    return edge_index, edge_weight

def markov_process_agg_sparse(data, eps, inflate, nlayers, row_normalization = True, debug = True):
    # sparse operations
    start = time.time()
    ei, ew = data.edge_index, data.edge_attr
    if ew is None or len(ew) == 0:
        ew = torch.ones(ei.shape[1])
    medge_index = []
    medge_weight = []
    medge_index.append(ei)
    medge_weight.append(ew)
    for i in range(nlayers-1):
        ei, ew = spspmm(ei, ew, ei, ew, len(data.x), len(data.x), len(data.x))
        ew = torch.pow(ew, inflate)
        # normalization of matrix
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'row')
        else:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'col')
        # pruning stage
        remaining_edge_idx = torch.nonzero(ew >= eps).flatten()
        ei = ei[:,remaining_edge_idx]
        ew = ew[remaining_edge_idx]
        # normalization
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'row')
        else:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'col')
        # store layer-wise edges
        medge_index.append(ei)
        medge_weight.append(ew)
        if debug:
            print("layer ", i+1, "(after sparsification) edge_index size:", ei.shape)
    if nlayers > len(medge_index):
        print("Use less number of layers for the given", eps, " threshold, maximum:", len(medge_index), "layers")
        sys.exit(1)
    end = time.time()
    if debug:
        print("Time required for sparse markov process:", end - start, "seconds")
    return (medge_index, medge_weight)
        

def markov_process_agg(data, eps, inflate, nlayers, row_normalization = True, keepmax = True, debug = True):
    start = time.time()
    A = to_dense_adj(edge_index = data.edge_index, batch = None, edge_attr = data.edge_attr, max_num_nodes = int(data.x.shape[0]))[0]
    (ei, ew) = dense_to_sparse(A)
    medge_index = []
    medge_weight = []
    medge_index.append(ei)
    medge_weight.append(ew)
    if debug:
        print("layer 0", "edge_index initial size:", data.edge_index.shape)
    AP = A.clone()
    for i in range(nlayers-1):
        A = torch.mm(A, A)
        A = torch.pow(A, inflate)
        (ei, ew) = dense_to_sparse(A)
        if debug:
            print("layer ", i+1, " (after mul and pow) edge_index size:", ei.shape)
        # normalization
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, A.shape[0], 'row')
        else:
            ei, ew = markov_normalization(ei, ew, A.shape[0], 'col')
        if keepmax:
            # sparsification threshold
            remaining_edge_idx = torch.nonzero(ew >= eps).flatten()
            ei = ei[:,remaining_edge_idx]
            ew = ew[remaining_edge_idx]
        else:
            # keep max value in each row of the adj matrix
            A = to_dense_adj(edge_index = ei, batch = None, edge_attr = ew, max_num_nodes = int(data.x.shape[0]))[0]
            for i in range(len(A)):
                idx = torch.nonzero(A[i] < eps).flatten()
                if len(idx) == len(A[i]):
                    idmax = torch.argmax(A[i])
                    valmax = torch.max(A[i])
                    A[i, idx] = 0
                    A[i, idmax] = valmax
                else:
                    A[i, idx] = 0
            (ei, ew) = dense_to_sparse(A)
        if ei.shape[1] < 1:
            print("No more edges..! stopping after ", i, "layers")
            break
        #normalization
        if row_normalization:
            edge_index2, edge_weight2 = markov_normalization(ei, ew, A.shape[0], 'row')
        else:
            edge_index2, edge_weight2 = markov_normalization(ei, ew, A.shape[0], 'col')
        A = to_dense_adj(edge_index = edge_index2, batch = None, edge_attr = edge_weight2, max_num_nodes = int(data.x.shape[0]))[0]
        if debug:
            print("layer ", i+1, "(after sparsification) edge_index size:", edge_index2.shape)
        medge_index.append(edge_index2)
        medge_weight.append(edge_weight2)
    if nlayers > len(medge_index):
        print("Use less number of layers for the given", eps, " threshold, maximum:", len(medge_index), "layers")
        sys.exit(1)
    end = time.time()
    if debug:
        print("Time required for dense markov process:", end - start, "seconds")
    return (medge_index, medge_weight)


def markov_process_disj_sparse(data, eps, inflate, nlayers, row_normalization = True, keepmax = True, debug = True):
    start = time.time()
    ei, ew = data.edge_index, data.edge_attr
    if ew is None or len(ew) == 0:
        ew = torch.ones(ei.shape[1])
    medge_index = []
    medge_weight = []
    medge_index.append(ei)
    medge_weight.append(ew)
    prev_edge_index = ei
    # markov process converges less than 30 iterations for the tested graphs
    for i in range(30):
        # sparse matrix-matrix multiplication
        ei, ew = spspmm(ei, ew, ei, ew, len(data.x), len(data.x), len(data.x))
        ew = torch.pow(ew, inflate)
        # normalization of matrix
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'row')
        else:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'col')
        # pruning stage
        remaining_edge_idx = torch.nonzero(ew >= eps).flatten()
        ei = ei[:,remaining_edge_idx]
        ew = ew[remaining_edge_idx]
        # normalization
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'row')
        else:
            ei, ew = markov_normalization(ei, ew, len(data.x), 'col')
        # store layer-wise edges
        medge_index.append(ei)
        medge_weight.append(ew)
        if debug:
            print("layer ", i+1, "(after sparsification) edge_index size:", ei.shape)
        if ei[0].shape == prev_edge_index[0].shape:
            print("early stopping markov process due to converged number of edges.")
            break
        prev_edge_index = ei

    if nlayers > len(medge_index):
        print("Use less number of layers for the given", eps, " threshold, maximum:", len(medge_index), "layers")
        sys.exit(1)
    # taking l matrices from k markov matrices
    mei = []
    mew = []
    step = len(medge_index) // nlayers
    idx = step
    mei.append(medge_index[0])
    mew.append(medge_weight[0])
    for l in range(1, nlayers-1):
        mei.append(medge_index[idx])
        mew.append(medge_weight[idx])
        idx += step
    mei.append(medge_index[len(medge_index)-1])
    mew.append(medge_weight[len(medge_index)-1])
    end = time.time()
    if debug:
        print("Time required for sparse markov process:", end - start, "seconds")
    return (mei, mew)


def markov_process_disj(data, eps, inflate, nlayers, row_normalization = True, keepmax = True, debug = True):
    start = time.time()
    A = to_dense_adj(edge_index = data.edge_index, batch = None, edge_attr = data.edge_attr, max_num_nodes = int(data.x.shape[0]))[0]
    (ei, ew) = dense_to_sparse(A)
    medge_index = []
    medge_weight = []
    medge_index.append(ei)
    medge_weight.append(ew)
    if debug:
        print("layer 0", "edge_index initial size:", data.edge_index.shape)
    AP = A.clone()
    prev_edge_index = ei
    for i in range(30):
        A = torch.mm(A, A)
        A = torch.pow(A, inflate)
        (ei, ew) = dense_to_sparse(A)
        if debug:
            print("layer ", i+1, " (after mul and pow) edge_index size:", ei.shape)
        # normalization
        if row_normalization:
            ei, ew = markov_normalization(ei, ew, A.shape[0], 'row')
        else:
            ei, ew = markov_normalization(ei, ew, A.shape[0], 'col')
        if keepmax:
            # sparsification threshold
            remaining_edge_idx = torch.nonzero(ew >= eps).flatten()
            ei = ei[:,remaining_edge_idx]
            ew = ew[remaining_edge_idx]
        else:
            # keep max value in each row of the adj matrix
            A = to_dense_adj(edge_index = ei, batch = None, edge_attr = ew, max_num_nodes = int(data.x.shape[0]))[0]
            for i in range(len(A)):
                idx = torch.nonzero(A[i] < eps).flatten()
                if len(idx) == len(A[i]):
                    idmax = torch.argmax(A[i])
                    valmax = torch.max(A[i])
                    A[i, idx] = 0
                    A[i, idmax] = valmax
                else:
                    A[i, idx] = 0
            (ei, ew) = dense_to_sparse(A)
        if ei.shape[1] < 1:
            print("No more edges..! stopping after ", i, "layers")
            break
        #normalization
        if row_normalization:
            edge_index2, edge_weight2 = markov_normalization(ei, ew, A.shape[0], 'row')
        else:
            edge_index2, edge_weight2 = markov_normalization(ei, ew, A.shape[0], 'col')
        A = to_dense_adj(edge_index = edge_index2, batch = None, edge_attr = edge_weight2, max_num_nodes = int(data.x.shape[0]))[0]
        if debug:
            print("layer ", i+1, "(after sparsification) edge_index size:", edge_index2.shape)
        medge_index.append(edge_index2)
        medge_weight.append(edge_weight2)
        if edge_index2[0].shape == prev_edge_index[0].shape:
            print("early stopping markov process due to converged number of edges.")
            break
        prev_edge_index = edge_index2
    if nlayers > len(medge_index):
        print("Use less number of layers for the given", eps, " threshold, maximum:", len(medge_index), "layers")
        sys.exit(1)
    mei = []
    mew = []
    step = len(medge_index) // nlayers
    idx = step
    mei.append(medge_index[0])
    mew.append(medge_weight[0])
    for l in range(1, nlayers-1):
        mei.append(medge_index[idx])
        mew.append(medge_weight[idx])
        idx += step
    mei.append(medge_index[len(medge_index)-1])
    mew.append(medge_weight[len(medge_index)-1])
    end = time.time()
    if debug:
        print("Time required for dense markov process:", end - start, "seconds")
    return (mei, mew)
