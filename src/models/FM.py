import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob, l2):
        super(FM, self).__init__()
        """
            num_features: number of features,
            num_factors: number of hidden factors,
            batch_norm: bool type, whether to use batch norm or not,
            drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.l2 = l2

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        fm_modules = []
        if self.batch_norm:
            fm_modules.append(nn.BatchNorm1d(num_factors))
        fm_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*fm_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)

    def l2_regularization(self):
        return self.l2 * self.embeddings.weight.norm()
