import torch
import torch.nn as nn
from layers import MLP


class NFM(nn.Module):
    def __init__(self, num_features, num_factors,
                 act_function, layers, batch_norm, drop_prob, l2, pre_trained_FM=None):
        super(NFM, self).__init__()
        """
            num_features: number of features,
            num_factors: number of hidden factors,
            act_function: activation function for MLP layer,
            layers: list of dimension of deep layers,
            batch_norm: bool type, whether to use batch norm or not,
            drop_prob: list of the dropout rate for FM and MLP,
            pre_trained_FM: the pre-trained FM weights.
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.l2 = l2
        self.pre_trained_FM = pre_trained_FM

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

        fm_modules = []
        if self.batch_norm:
            fm_modules.append(nn.BatchNorm1d(num_factors))
        fm_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*fm_modules)

        # deep layers

        self.deep_layers = MLP(num_factors, self.layers, self.drop_prob, self.act_function, self.batch_norm)

        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """ Try to mimic the original weight initialization. """
        if self.pre_trained_FM:
            self.embeddings.weight.data.copy_(self.pre_trained_FM.embeddings.weight)
            self.biases.weight.data.copy_(self.pre_trained_FM.biases.weight)
            self.global_bias.data.copy_(self.pre_trained_FM.global_bias)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # init deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        # Embedding Layer
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=2)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)
        fm = 0.5 * (sum_square_embed - square_sum_embed)
        fm = self.FM_layers(fm)

        # Hidden Layer
        if self.layers:  # have deep layers
            fm = self.deep_layers(fm)

        # Prediction Score
        fm = self.prediction(fm)
        feature_bias = self.biases(features)  # bias addition
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        fm = fm + feature_bias + self.global_bias
        return fm.view(-1)

    def l2_regularization(self):
        return self.l2 * self.embeddings.weight.norm()
