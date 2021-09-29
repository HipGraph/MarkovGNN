import os.path as osp
import argparse
import sys
import numpy as np
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch
from networkx.algorithms import community
import networkx as nx

def readInput3f(inputf, labelf, featuref, oneIndexed = False, onelabeled = False, debug = True):
    # inputf: input file name with path
    inpf = open(inputf)
    edgelist = []
    for line in inpf.readlines():
        tokens = line.strip().split()
        src = int(tokens[0])
        dst = int(tokens[1])
        # substract 1 if vertex ids start from 1 or, oneIndexed
        if oneIndexed:
            src -= 1
            dst -= 1
        edgelist.append([src, dst])
    inpf.close()
    # labelf: label file name with path
    labels = dict()
    lfile = open(labelf)
    if debug:
        print("Assuming label ids are sorted based on vertex ids!")
    i = 0
    for line in lfile.readlines():
        tokens = line.strip().split()
        # skip empty lines
        if len(tokens) == 0:
            continue
        lab = int(line)
        # substract 1 if label ids start from 1 or, onelabeled
        if onelabeled:
            lab = lab - 1
        labels[i] = lab
        i += 1
    lfile.close()

    edgelist.sort(key = lambda x: (x[0], x[1]))
    nodes = torch.tensor([labels[i] for i in range(len(labels))])
    numberofclasses = len(set(list(labels.values())))
    # check how many samples are available for different classes
    classdict = dict()
    iterableclass = list(set(list(labels.values())))
    for c in iterableclass:
        classdict[c] = 0
    for i in range(len(labels)):
        classdict[labels[i]] += 1
    if debug:
        print("Per-label available samples.")
        for key in classdict:
            print("label id:", key, "#of samples:", classdict[key])
    # end checking of samples per class
    # load features from featuref file
    feat = np.loadtxt(featuref, dtype=float)
    features = torch.FloatTensor(feat)
    numberoffeatures = features.shape[1]
    a = []
    b = []
    for e in edgelist:
        a.append(e[0])
        b.append(e[1])
    edgelist = torch.tensor([a,b])
    # create dataset using PyG data class
    data = Data(x = features, edge_index = edgelist, y = nodes)
    # 70% train, 10% validation, 20% test
    trainp = int(len(labels) * 0.70)
    valp = int(len(labels) * 0.10)
    # train 70% dataset
    trainm = [True if i < trainp else False for i in range(len(labels))]
    # enable it if want to train 30 nodes for each class
    if False:
        classdict = dict()
        iterableclass = list(set(list(labels.values())))
        for c in iterableclass:
            classdict[c] = 0
        trainm = []
        for i in range(len(labels)):
            if classdict[labels[i]] < 30:
                trainm.append(True)
                classdict[labels[i]] += 1
            else:
                trainm.append(False)
    if debug:
        print("Total #of samples for training:", sum(trainm))
    valm = [True if i >= trainp and i < trainp + valp else False for i in range(len(labels))]
    testm = [True if i >= trainp + valp else False for i in range(len(labels))]
    data.train_mask = torch.tensor(trainm, dtype=torch.bool)
    data.val_mask = torch.tensor(valm, dtype=torch.bool)
    data.test_mask = torch.tensor(testm, dtype=torch.bool)
    return data

def readInput2f(inputf, labelf, oneIndexed = False, onelabeled = False, debug = True):
    # inputf: input file name with path
    inpf = open(inputf)
    edgelist = []
    for line in inpf.readlines():
        tokens = line.strip().split()
        src = int(tokens[0])
        dst = int(tokens[1])
        if oneIndexed == 1:
            src -= 1
            dst -= 1
        edgelist.append([src, dst])
    inpf.close()
    # labelf: label file name with path
    labels = dict()
    lfile = open(labelf)
    for line in lfile.readlines():
        tokens = line.strip().split()
        if len(tokens) == 0:
            continue
        node = int(tokens[0])
        if oneIndexed:
            node = node - 1
        lab = int(tokens[1])
        if onelabeled:
            lab = lab - 1
        labels[node] = lab
    lfile.close()
    edgelist.sort(key = lambda x: (x[0], x[1]))
    nodes = torch.tensor([labels[i] for i in range(len(labels))])
    numberofclasses = len(set(list(labels.values())))
    numberoffeatures = len(labels)
    # check how many samples are available for different classes
    classdict = dict()
    iterableclass = list(set(list(labels.values())))
    for c in iterableclass:
        classdict[c] = 0
    for i in range(len(labels)):
        classdict[labels[i]] += 1
    if debug:
        for key in classdict:
            print("label id:", key, "#of samples:", classdict[key])
    #End checking of samples per class

    a = []
    b = []
    for e in edgelist:
        a.append(e[0])
        b.append(e[1])
    edgelist = torch.tensor([a,b])
    seq_nodes = torch.tensor([i for i in range(len(nodes))])
    one_hot = torch.nn.functional.one_hot(seq_nodes).float()
    data = Data(x = one_hot, edge_index = edgelist, y = nodes)
    trainp = int(len(labels) * 0.70)
    valp = int(len(labels) * 0.10)
    # train 70% dataset
    trainm = [True if i < trainp else False for i in range(len(labels))]
    # enable it if want to train 30 nodes for each class
    if False:
        classdict = dict()
        iterableclass = list(set(list(labels.values())))
        for c in iterableclass:
            classdict[c] = 0
        trainm = []
        for i in range(len(labels)):
            if classdict[labels[i]] < 30:
                trainm.append(True)
                classdict[labels[i]] += 1
            else:
                trainm.append(False)
    if debug:
        print("Total #of samples for training:", sum(trainm))
    valm = [True if i >= trainp and i < trainp + valp else False for i in range(len(labels))]
    testm = [True if i >= trainp + valp else False for i in range(len(labels))]
    data.train_mask = torch.tensor(trainm, dtype=torch.bool)
    data.val_mask = torch.tensor(valm, dtype=torch.bool)
    data.test_mask = torch.tensor(testm, dtype=torch.bool)
    return data

def loadPyGDataset(dataset_name = 'Cora'):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
    dataset = Planetoid(path, dataset_name, num_train_per_class=30, transform=T.NormalizeFeatures())
    data = dataset[0]
    return data

# compute homophily
def computeHomophily(data, ei = None):
    if ei is None:
        edges = data.edge_index.t()
    else:
        edges = ei.t()
    nominator = 0
    for edge in edges:
        nominator += data.y[edge[0]] == data.y[edge[1]]
    return nominator / len(edges)

# compute community mixing
def mixingCommunityScore(data):
    G = nx.Graph(data.edge_index.t().tolist())
    comm = community.greedy_modularity_communities(G)
    gd = dict()
    for com in range(len(comm)):
        for node in list(comm[com]):
            gd[node] = com
    count = 0
    for edge in data.edge_index.t():
        count += gd[edge[0].item()] != gd[edge[1].item()]
    return count / len(data.edge_index.t())

# compute new edges percentage
def newEdges(data, edges):
    count = 0
    for edge in edges.t():
        if edge not in data.edge_index.t():
            count += 1
    return 100.0 * count / len(edges.t())
