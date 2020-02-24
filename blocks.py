# coding: utf-8

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

from hyperparameters import hp


class CharEmbed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(CharEmbed, self).__init__()

        self.embed = nn.utils.weight_norm(nn.Embedding(num_embeddings, embedding_dim, padding_idx))

    def forward(self, x):
        output = self.embed(x)
        return output.transpose(1, 2)

class Conv1d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride,padding,
                dilation,groups,bias):
        super(Conv1d, self).__init__()
        if padding == 'same':
            padding = ((kernel_size - 1) * dilation + 1) // 2
        elif padding == 'custom':
            padding = (kernel_size - 1) * dilation

        self.conv = nn.utils.weight_norm(nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias))

    def forward(self, x):
        out = self.conv(x)

        return out[:, :, :x.size(2)]


class LayerNorm(nn.Module):
    r"""
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
                    \times \ldots \times \text{normalized_shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension with that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Layer Normalization`: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class HighwayNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride,
                 padding, dilation, groups, bias):

        super(HighwayNetwork, self).__init__()
        if padding == 'same':
            padding = ((kernel - 1) * dilation + 1) // 2
        elif padding == 'custom':
            padding = (kernel - 1) * dilation
        #use custom convolution layer
        self.conv1d = Conv1d(in_channels=in_channels,
                              out_channels=2*out_channels,
                              kernel_size=kernel,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)


    def forward(self, x):

        _x = self.conv1d(x)

        H1, H2 = torch.chunk(_x, 2, 1)

        #H1 = LayerNorm(_x.size()[1:])(H1)
        #H2 = LayerNorm(_x.size()[1:])(H2)

        H1 = F.sigmoid(H1)
        #H2 = F.relu(H2)

        output = H1*H2 + (1.0 - H1) * x

        return output

class HighwayNetworkAlter(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(HighwayNetworkAlter, self).__init__()

        self.linear1 = nn.Linear(in_features=in_features,
                                out_features=out_features,
                                bias=bias)

        self.linear2 = nn.Linear(in_features=in_features,
                                out_features=out_features,
                                bias=bias)

        def forward(self, x):
            H1 = self.linear1(x)
            H1 = F.relu(H1)

            H2 = self.linear2(x)
            H2 = F.sigmoid(H2)

            out = H1*H2 + x*(1.0 - H2)

            return out

class Highway(nn.Module):
    def __init__(self, size, num_layers, non_lin_func):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.H = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.T = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.non_lin_func = non_lin_func

    def forward(self, x):

        for layer in range(self.num_layers):
            T = F.sigmoid(self.T[layer](x)) #gate

            H = self.non_lin_func(self.H[layer](x)) #non-linear transformation

            x = H * T + x * (1 - T)

        return x

class Deconv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Deconv1d, self).__init__()
        if padding == "same":
            padding = max(0, (kernel_size - 2) // 2)
        self.deconv = nn.ConvTranspose1d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=bias)

        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.batch_norm(x)
        x = F.relu(x)

        return x
