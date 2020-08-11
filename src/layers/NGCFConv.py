import torch
from torch import nn
from torch.nn import init
import dgl
from dgl.base import DGLError
from .activation import activation_layer


class NGCFConv(nn.Module):
    r"""
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``LeakyReLu``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """

    def __init__(self, in_feats, out_feats,
                 dropout, normalized,
                 activation='LeakyReLu', bias=True):
        super(NGCFConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.dropout_msg = nn.Dropout(dropout)
        self.normalized = normalized
        self.activation = activation_layer(activation)

        self.weight1 = nn.Parameter(torch.zeros(in_feats, out_feats))  # (d, e)
        self.weight2 = nn.Parameter(torch.zeros(in_feats, out_feats))  # (d, e)
        self.weight_list = [self.weight1, self.weight2]

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_feats))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight1 is not None:
            init.xavier_uniform_(self.weight1)
        if self.weight2 is not None:
            init.xavier_uniform_(self.weight2)
        if self.bias is not None:
            init.zeros_(self.bias)

    def message(self, edges):
        # W_1@e_i
        interaction1 = edges.src['h'] @ self.weight1
        # W_2@(e_i*e_u)
        interaction2 = (edges.src['h'] * edges.dst['h']) @ self.weight2
        interaction = interaction1 + interaction2  # shape: (|E|, e)
        # (|N_u||N_i|)^0.5
        weight_decay = (edges.src['deg'] ** 0.5 * edges.dst['deg'] ** 0.5).unsqueeze(dim=1)
        interaction = interaction / weight_decay  # normalized
        return {'msg': interaction}

    def aggregation(self, nodes):
        # m_{u<-u} + \Sum m_{u<-i}
        embed = nodes.data['h'] @ self.weight1 + nodes.mailbox['msg'].sum(dim=1)  # shape: (*, e)

        return {'embed': embed}

    def forward(self, graph: dgl.DGLGraph, feat):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature

        Returns
        -------
        torch.Tensor
            The output feature
        """
        # add graph features
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.ndata['deg'] = graph.out_degrees().float().clamp(min=1)

        # message passing
        graph.update_all(message_func=self.message, reduce_func=self.aggregation)
        embeds = graph.ndata['embed']
        if self.bias is not None:
            embeds = embeds + self.bias
        if self.activation is not None:
            embeds = self.activation(embeds)
        # message dropout
        embeds = self.dropout_msg(embeds)
        if self.normalized:
            embeds = nn.functional.normalize(embeds, p=2, dim=1)
        return embeds
