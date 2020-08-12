import torch
from torch import nn
from dgl import DGLGraph
from layers import NGCFConv


class NGCF(nn.Module):
    def __init__(self, node_map, user_bought,
                 embedding_size, layers_dim, dropout_rate, normalized, l2,
                 activation='LeakyReLu'):
        super(NGCF, self).__init__()
        self.node_map = node_map
        self.user_bought = user_bought  # {user(int): [item(int), item(int), ...]}
        self.embedding_size = embedding_size
        self.layers_dim = layers_dim
        self.dropout = dropout_rate
        self.normalized = normalized
        self.l2 = l2
        self.activation = activation

        self.weight_list = []

        self.graph = self.construct_graph()
        # initial embeddings of each nodes in graph
        self.h0 = nn.Embedding(len(self.node_map), embedding_size)
        self.convs = nn.ModuleList()

        in_feats = self.embedding_size
        for i, dim in enumerate(layers_dim):
            self.convs.append(NGCFConv(in_feats, dim, self.dropout[i], self.normalized, self.activation))
            self.weight_list += self.convs[-1].weight_list
            in_feats = dim

        self.reset_parameters()

    def construct_graph(self):
        graph = DGLGraph()
        graph.add_nodes(len(self.node_map))  # |V| = N+M
        for user, item in enumerate(self.user_bought):
            graph.add_edges([user, item], [item, user])
        return graph

    def reset_parameters(self):
        # embeddings
        nn.init.normal_(self.h0.weight, 0.0, 0.01)

    def forward(self, features, feature_values):
        # -------------Embedding-------------
        users = list(map(lambda feature: self.node_map[int(feature[0])], features))  # (batch,)
        items = list(map(lambda feature: self.node_map[int(feature[1])], features))  # (batch,)

        h = [self.h0.weight]
        for conv in self.convs:
            h.append(conv(self.graph, h[-1]))
        h = torch.cat(h, dim=1)  # [(|V|, d^(1)), (|V|, d^(2)), ..., (|V|, d^(l))]

        # user_embeddings = torch.tensor([h[user] for user in users])
        # item_embeddings = torch.tensor([h[item] for item in items])
        user_embeddings = h[[users]]
        item_embeddings = h[[items]]

        return (user_embeddings * item_embeddings).sum(dim=1)

    def l2_regularization(self):
        l2_reg = 0
        for weight in self.weight_list:
            l2_reg += weight.norm()
        return self.l2 * l2_reg

