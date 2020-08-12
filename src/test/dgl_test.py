from layers import NGCFConv
import dgl
import torch
from torch import nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model = NGCFConv(3, 5, dropout=0.0, normalized=True)
    src = torch.tensor([0, 0, 0, 0, 0])
    dst = torch.tensor([1, 2, 3, 4, 5])
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    graph = dgl.DGLGraph((u, v))
    features = nn.Embedding(6, 3)
    embed = model(graph, features.weight)
    print(embed)

    graph_2 = dgl.DGLGraph(([6, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]))
    graph_2.add_nodes(1)
    # nx.draw(graph_2.to_networkx(), with_labels=True)
    # plt.show()
    features = nn.Embedding(8, 3)
    print(features.weight)
    embed = model(graph_2, features.weight)
    print(embed)
