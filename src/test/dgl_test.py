from layers import NGCFConv
import dgl
import torch
from torch import nn
import numpy as np
import models


if __name__ == '__main__':
    model = NGCFConv(3, 5, 'both', True, False, 'LeakyReLu')
    src = torch.tensor([0, 0, 0, 0, 0])
    dst = torch.tensor([1, 2, 3, 4, 5])
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    graph = dgl.DGLGraph((u, v))
    features = nn.Embedding(6, 3)
    prediction = model(graph, features.weight)
