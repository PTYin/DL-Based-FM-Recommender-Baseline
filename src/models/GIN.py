import torch
from torch import nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv, GraphConv
import warnings


class GIN(nn.Module):
    def __init__(self, feature_size, field_size,
                 embedding_size, deep_layers_dim, dropout_deep, deep_act, batch_norm, l2,
                 g, num_layers, in_dim, num_hidden, num_classes, heads, gat_act, feat_drop, attn_drop,
                 negative_slope, residual):
        super(GIN, self).__init__()
        warnings.warn("GIN is deprecated", DeprecationWarning)

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers_dim = deep_layers_dim
        self.dropout_deep = dropout_deep
        self.deep_act = deep_act
        self.batch_norm = batch_norm
        self.l2 = l2

        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_act = gat_act

        # Embedding
        self.embeddings = nn.Embedding(self.feature_size, self.embedding_size)

        # Assign features to nodes
        # self.g.ndata['feat'] = self.embeddings.weight[::self.field_size]

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.gat_act))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.gat_act))

        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], self.embedding_size, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features: torch.LongTensor, feature_values: torch.FloatTensor):

        # -------------Embedding-------------
        feature_embeddings = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=2)
        feature_embeddings = feature_embeddings * feature_values

        # -------------Graph Intention Discovery-------------
        h = feature_embeddings[:][0]  # always put item feature in the first
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(2)  # TODO
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(2)
        return logits