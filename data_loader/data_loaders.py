import os
from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import networkx as nx

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, test_split=0.2, num_workers=1, training=True):
        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_dir = data_dir
        ddi_df, ppi_df, dpi_df = self.load_data()

        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True)#, transform=trsfm
        # super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)

        # build graph
        # idx = dpi_ppi[1].tolist()
        # idx=np.array(idx)
        # idx_map = {j: i for i, j in enumerate(idx)}

        dpi_edges=dpi_df[['drugbank_id','Entrez']].values
        ppi_edges=ppi_df[['protein1','protein2']].values
        edges_unordered =np.vstack([dpi_edges,ppi_edges])
        graph = build_graph(edges_unordered)
        adj=nx.adjacency_matrix(graph)

        labels = ddi_df.iloc[:, -1]
        features = np.random.randint((0,1), size=(len(edges_unordered), 64))
        #features= nn.Embedding(len(edges_unordered), 64, max_norm=True)
        features = sp.csr_matrix(features)

        # edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
        #                 dtype=np.int32).reshape(edges_unordered.shape)
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                     shape=(labels.shape[0], labels.shape[0]),
        #                     dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # idx_train = range(140)
        # idx_val = range(200, 500)
        # idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        return adj, features, labels #, idx_train, idx_val, idx_test

    def load_data(self):
        ddi_df = pd.read_csv('./data/ddi.csv')
        ppi_df = pd.read_csv('./data/ppi.csv')
        dpi_df = pd.read_csv('./data/dpi.csv')

        return ddi_df, ppi_df, dpi_df

def build_graph(edges):
    tuples = [tuple(x) for x in edges]
    graph = nx.Graph()
    graph.add_edges_from(tuples)
    return graph

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
