import torch, random
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, MGCNConv 
from sklearn.cluster import KMeans
import networkx as nx
import community as comm

class MarkovGCNR(torch.nn.Module):
    def __init__(self, ndim, nlayers, ntargets, features, edges, weights = None, droprate = 0.5, useleakyrelu = False, alpha = 0.5, addbias = True):
        super(MarkovGCNR, self).__init__()
        self.convs = []
        self.ndim = ndim
        self.nlayers = nlayers
        self.edges = edges
        self.weights = weights
        self.ntargets = ntargets
        self.features = features
        self.droprate = droprate
        self.useleakyrelu = useleakyrelu
        self.alpha = alpha
        self.convs.append(GCNConv(self.features.shape[1], self.ndim, cached=True, bias = addbias))
        for l in range(nlayers-2):
            self.convs.append(GCNConv(self.ndim, self.ndim, cached=True, bias = addbias))
        self.convs.append(GCNConv(self.ndim, self.ntargets, cached=True, bias = addbias))

    def forward(self):
        assert len(self.edges) == self.nlayers
        x = F.dropout(self.features, p=self.droprate, training=self.training)
        x = self.convs[0](x, self.edges[0], None)
        x_prev = x
        for l in range(1, self.nlayers):
            if self.useleakyrelu:
                x = F.leaky_relu(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, p=self.droprate, training=self.training)
            if l < self.nlayers - 1:
                # residual connection with first layer
                x = self.alpha * self.convs[l](x, self.edges[l], self.weights[l]) + (1-self.alpha) * x_prev
                #x = self.convs[l](self.alpha * x + (1-self.alpha) * x_prev, medge_index[l], medge_weight[l])
            else:
                x = self.convs[l](x, self.edges[l], self.weights[l])
        return F.log_softmax(x, dim=1)

    def inference(self):
        # inference considering 2-layered network
        x = self.convs[0](self.features, self.edges[0])
        xs = F.relu(x).cpu().detach().numpy()
        xs = xs.round(decimals=5)
        return xs

class GCN(torch.nn.Module):
    def __init__(self, ndim, nlayers, ntargets, features, edges, weights = None, droprate = 0.5, alpha = 0.5, addbias = True):
        super(GCN, self).__init__()
        self.convs = []
        self.ndim = ndim
        self.nlayers = 2
        self.edges = edges
        self.weights = weights
        self.ntargets = ntargets
        self.features = features
        self.droprate = droprate
        self.convs.append(GCNConv(self.features.shape[1], self.ndim, cached=True, bias = addbias))
        self.convs.append(GCNConv(self.ndim, self.ntargets, cached=True, bias = addbias))

    def forward(self):
        x = F.dropout(self.features, p=self.droprate, training=self.training)
        x = self.convs[0](x, self.edges, None)
        x = F.relu(x)
        x = F.dropout(x, p=self.droprate)
        x = self.convs[1](x, self.edges, self.weights)
        return F.log_softmax(x, dim=1)
