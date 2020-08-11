import torch as th
from torch import nn
from torch.nn import init
import dgl
from dgl import function as fn
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
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=False,
                 activation=None):
        super(NGCFConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight1 = nn.Parameter(th.Tensor(in_feats, out_feats))
            self.weight2 = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation_layer(activation)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight1 is not None:
            init.xavier_uniform_(self.weight1)
        if self.weight2 is not None:
            init.xavier_uniform_(self.weight2)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph: dgl.DGLGraph, feat, weight=None):
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
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        # normalize
        if self._norm == 'both':
            degrees = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = th.pow(degrees, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            feat = feat * norm

        # name weight
        weight1 = self.weight1
        weight2 = self.weight2

        # def message_func(edges):
        #     dst_data = edges.dst['h']
        #     src_data = edges.src['h']
        #     return_data = dst_data * src_data
        #     return {'inner_multi': return_data}

        # prob
        graph.srcdata['h'] = feat
        graph.update_all(fn.copy_u('h', out='copy'),
                         fn.sum(msg='copy', out='copy_sum'))
        graph.update_all(lambda edges: {'inner_multi': edges.src['h'] * edges.dst['h']},
                         fn.sum(msg='inner_multi', out='inner_multi_sum'))
        rst1 = th.matmul(graph.dstdata['copy_sum'], weight1)
        rst2 = th.matmul(graph.dstdata['inner_multi_sum'], weight2)
        rst = rst1 + rst2

        if self._norm != 'none':
            degrees = graph.in_degrees().float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degrees, -0.5)
            else:
                norm = 1.0 / degrees
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)
