import torch
import torch.nn as nn
import torch.nn.functional as F


class AFM(nn.Module):
    def __init__(self, num_features, num_factors,
                 attention, valid_dim, drop_prob, pretrain_FM):
        super(AFM, self).__init__()
        """
            num_features: number of features,
            num_factors: number of hidden factors,
            attention: flag whether or not use attention,
            valid_dim: the dimension of valid values,
            drop_prob: dropout rate,
            pretrain_FM: the pre-trained FM weights.
        """
        self.num_features = num_features
        self.num_factors = num_factors[1]
        self.num_att_factors = num_factors[0]
        self.attention = attention
        self.valid_dim = valid_dim
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM

        self.embeddings = nn.Embedding(self.num_features, self.num_factors)
        self.biases = nn.Embedding(self.num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        self.att_layer = nn.Sequential(
            nn.Linear(self.num_factors, self.num_att_factors),
            nn.ReLU(),
            nn.Linear(self.num_att_factors, 1, bias=False),
            nn.Softmax(dim=1)
        )
        # self.att_h = nn.Parameter(torch.randn(self.num_att_factors))
        self.prediction = nn.Linear(self.num_factors, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(
                self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(
                self.pretrain_FM.biases.weight)
            self.bias_.data.copy_(self.pretrain_FM.bias_)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        nn.init.xavier_normal_(self.att_layer[0].weight)
        nn.init.normal_(self.att_layer[2].weight, std=1.0)
        nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        # input processing
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Compute interactions first
        ele_product_list = []
        for i in range(0, self.valid_dim):
            for j in range(i + 1, self.valid_dim):
                ele_product_list.append(
                    nonzero_embed[:, i, :] * nonzero_embed[:, j, :])
        ele_product = torch.stack(ele_product_list, dim=1)

        # attention weights computation and multiplication
        if self.attention:
            att_out = self.att_layer(ele_product)
            # att_out = (self.att_h * att_out).sum(dim=2, keepdim=True)
            # att_out = F.softmax(att_out, dim=1)
            AFM = (att_out * ele_product).sum(dim=1)
        else:
            AFM = ele_product.sum(dim=1)
        AFM = F.dropout(AFM, p=self.drop_prob)

        # FM model
        AFM = self.prediction(AFM)
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        AFM = AFM + feature_bias + self.bias_
        return AFM.view(-1)