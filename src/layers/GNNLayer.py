import torch
from torch import nn


class GNNLayer(nn.Module):

    def __init__(self, inF, outF):

        super(GNNLayer, self).__init__()
        self.inF = inF
        self.outF = outF
        self.linear = torch.nn.Linear(in_features=inF, out_features=outF)
        self.interActTransform = torch.nn.Linear(in_features=inF, out_features=outF)

    def forward(self, l, self_loop, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # L = D^-(1/2)(A)D^(-1/2) # 拉普拉斯矩阵
        l1 = l + self_loop
        l2 = l
        inter_feature = torch.sparse.mm(l2, features)
        inter_feature = torch.mul(inter_feature, features)

        inter_part1 = self.linear(torch.sparse.mm(l1, features))
        inter_part2 = self.interActTransform(torch.sparse.mm(l2, inter_feature))

        return inter_part1+inter_part2
