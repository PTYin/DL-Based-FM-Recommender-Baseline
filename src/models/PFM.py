import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from numpy.random import RandomState


class PMF(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(PMF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.random_state = RandomState(2020)

        self.embed_user.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(user_num, factor_num)).float()
        self.embed_item.weight.data = torch.from_numpy(
            0.1 * self.random_state.rand(item_num, factor_num)).float()

    def forward(self, user, item):
        user = self.embed_user(user)
        item = self.embed_item(item)

        prediction = (user * item).sum(dim=-1)
        return prediction
