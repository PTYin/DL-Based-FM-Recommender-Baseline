import torch as th
import numpy as np
import scipy.sparse as spp
import dgl
import networkx as nx
import matplotlib.pyplot as plt

# Create a star graph from a pair of arrays (using ``numpy.array`` works too).
u = th.tensor([0, 0, 0, 0, 0])
v = th.tensor([1, 2, 3, 4, 5])
star1 = dgl.DGLGraph((u, v))

# Create the same graph in one go! Essentially, if one of the arrays is a scalar,
# the value is automatically broadcasted to match the length of the other array
# -- a feature called *edge broadcasting*.
start2 = dgl.DGLGraph((0, v))

# Create the same graph from a scipy sparse matrix (using ``scipy.sparse.csr_matrix`` works too).
adj = spp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
star3 = dgl.DGLGraph(adj)

nx.draw(star1.to_networkx(), with_labels=True)
plt.show()
