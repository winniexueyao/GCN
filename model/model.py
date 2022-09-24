import torch
from torch import nn
from layers import GraphConvolution
from torch.nn import functional as F
from base import BaseModel
import numpy as np


class MnistModel(BaseModel):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, nfeat,  nclass): 
        super().__init__()
        # self.args = args
        self.feature_number=nfeat
        self.class_number = nclass
        self.gc = GraphConvolution(self.nfeat, self.nclass)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
                latent_features: latent representations of nodes
        """ 
        
        x = F.relu(self.gc(features, normalized_adjacency_matrix))
        x=np.inner(x[0], x[1])#?
        return F.log_softmax(x, dim=1)
