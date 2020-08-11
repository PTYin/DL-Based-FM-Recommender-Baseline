import torch
from torch import nn
from .activation import activation_layer


class MLP(nn.Module):
    def __init__(self, in_dim, layers_dim: list, dropout_rate: list, act_function, batch_norm: bool):
        super(MLP, self).__init__()

        self.weight_list = []

        mlp_module = []
        # in_dim = self.field_size * self.embedding_size
        for i in range(len(layers_dim)):
            out_dim = layers_dim[i]
            mlp_module.append(nn.Linear(in_dim, out_dim))
            self.weight_list.append(mlp_module[-1].weight)
            in_dim = out_dim
            if batch_norm:
                mlp_module.append(nn.BatchNorm1d(out_dim))

            mlp_module.append(activation_layer(act_function))

            mlp_module.append(nn.Dropout(dropout_rate[i]))

        self.layers = nn.Sequential(*mlp_module)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, embeddings):
        return self.layers(embeddings)
