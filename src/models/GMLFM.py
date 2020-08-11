import torch
import torch.nn as nn
import torch.nn.functional as F


class GMLFM(nn.Module):
    def __init__(self, num_features, num_factors,
                 act_function, use_product, num_layers, drop_prob):
        super(GMLFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        use_product: bool type, whether to use the former product,
        num_layers: number of of deep layers,
        drop_prob: dropout rate in deep layers.
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.act_function = act_function
        self.use_product = use_product
        self.num_layers = num_layers
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        self.w_h = nn.Parameter(torch.randn((num_factors,)))
        self.relation = nn.Parameter(torch.randn((num_factors, num_factors)))

        MLP_module = []
        for l in range(num_layers):
            MLP_module.append(nn.Dropout(drop_prob))
            MLP_module.append(nn.Linear(num_factors, num_factors))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())
        self.MLP_layers = nn.Sequential(*MLP_module)

        # init weights
        nn.init.normal_(self.w_h, std=0.01)
        nn.init.normal_(self.relation, std=0.01)
        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                m.bias.data.zero_()

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        embed_deep = self.MLP_layers(nonzero_embed)

        # relation = self.relation.transpose(-1, -2).mm(self.relation)
        relation = torch.ones(self.num_factors).diagflat().cuda()

        if self.use_product:
            fm1 = (nonzero_embed.sum(dim=1) *  # b*k
                   (self.w_h * nonzero_embed * (embed_deep *  # b*n*k
                                                embed_deep.matmul(relation)).sum(dim=-1, keepdim=True)
                    ).sum(dim=1)).sum(dim=-1)  # b

            fm2 = - ((nonzero_embed * self.w_h) * embed_deep.matmul(  # b*n*k
                relation).matmul((embed_deep.unsqueeze(dim=-2) *  # b*n*1*k
                                  nonzero_embed.unsqueeze(dim=-1)).sum(dim=1))  # b*k*k
                     ).sum(dim=-1).sum(dim=-1)  # b

        else:
            fm1 = (embed_deep * embed_deep.matmul(  # b*n*k
                relation)).sum(dim=-1).sum(dim=-1) * embed_deep.size()[1]

            fm2 = - (embed_deep.sum(dim=1) *  # b*k
                     embed_deep.sum(dim=1).matmul(relation)).sum(dim=-1)  # b

        # bias addition
        feature_bias = self.biases(features).sum(dim=1).sum(dim=1)
        fm = fm1 + fm2 + feature_bias + self.bias_
        return fm.view(-1)
