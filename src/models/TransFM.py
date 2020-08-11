import torch
import torch.nn as nn


class TransFM(nn.Module):
    def __init__(self, num_features, num_factors):
        super(TransFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors.
        """
        self.num_features = num_features
        self.num_factors = num_factors

        self.embeddings_e = nn.Embedding(num_features, num_factors)
        self.embeddings_t = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = torch.tensor([0.0], requires_grad=True).cuda()

        nn.init.uniform_(self.embeddings_e.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.embeddings_t.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.biases.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.bias_, a=-0.1, b=0.1)

    def forward(self, features, values):
        embed = self.embeddings_e(features)
        trans = self.embeddings_t(features)
        values = values

        fm1 = 0.5 * ((embed * embed).sum(dim=-1) * values).sum(dim=-1) * values.sum(dim=-1)
        fm2 = 0.5 * ((embed * trans).sum(dim=-1) * values).sum(dim=-1) * values.sum(dim=-1)
        fm3 = 0.5 * ((embed * embed).sum(dim=-1) * values).sum(dim=-1) * values.sum(dim=-1)
        fm4 = ((embed * trans).sum(dim=-1) * values).sum(dim=-1) * values.sum(dim=-1)
        fm5 = - ((embed * values.unsqueeze(dim=-1)).sum(dim=1) \
                 * (embed * values.unsqueeze(dim=-1)).sum(dim=1)).sum(dim=-1)
        fm6 = - ((embed * values.unsqueeze(dim=-1)).sum(dim=1) \
                 * (trans * values.unsqueeze(dim=-1)).sum(dim=1)).sum(dim=-1)
        fm7 = -0.5 * ((embed * trans).sum(dim=-1) * values.pow(2)).sum(dim=-1)

        fm = fm1 + fm2 + fm3 + fm4 + fm5 + fm6 + fm7

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias.squeeze(dim=-1) * values).sum(dim=1)
        fm = fm + feature_bias + self.bias_
        return fm.view(-1)
