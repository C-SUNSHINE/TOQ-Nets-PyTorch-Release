#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : stgcn.py
# Source : Modified from https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

from .stgcn_graph import Graph


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 binary_input_dim=0,
                 binary_output_dim=0,
                 use_max=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.binary_input_dim = binary_input_dim
        self.binary_output_dim = binary_output_dim
        self.use_max = use_max
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        if binary_input_dim > 0:
            assert binary_output_dim > 0
            self.relation_propagator = nn.Sequential(
                nn.Linear(out_channels + binary_input_dim, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
        else:
            self.relation_propagator = None

    def forward(self, x, A, binary_input=None):
        if A.size(0) == 1:
            A = A.repeat(self.kernel_size, 1, 1)
        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        n, kc, t, v = x.size()
        ks = self.kernel_size
        oc = kc // ks

        if self.relation_propagator is not None:
            # print(binary_input.size())
            x = x.view(n, ks, oc, t, v).permute(0, 1, 3, 4, 2).reshape(n * ks, t, v, oc).repeat(1, 1, v, 1)
            binary_input = binary_input.repeat(1, ks, 1, 1, 1).view(n * ks, t, v * v, self.binary_input_dim)
            res = self.relation_propagator(
                torch.cat([
                    x.view(-1, oc),
                    binary_input.view(-1, self.binary_input_dim)
                ], 1)).view(n, ks, t, v, v, -1)
            res = torch.einsum('nktvwc,kvw->nctv', (res, A))
            return res.contiguous(), A
        # print(x.mean())

        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        if self.use_max:
            x = torch.einsum('nkctv,kvw->nctwv', (x, A)).max(dim=4)[0]
        else:
            x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # print(x.mean())
        # input()

        return x.contiguous(), A


class STGCN_Layer(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 binary_input_dim=0,
                 binary_output_dim=0,
                 max_gcn=False,
                 **kwargs):
        super().__init__()

        assert len(kernel_size) >= 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        t_kernel_size = kernel_size[2] if len(kernel_size) > 2 else 1
        t_padding = (t_kernel_size - 1) // 2

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1],
                                         t_kernel_size=t_kernel_size, t_padding=t_padding,
                                         binary_input_dim=binary_input_dim,
                                         binary_output_dim=binary_output_dim,
                                         use_max=max_gcn)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        if binary_input_dim > 0:
            if stride == 1:
                self.relation_residual = lambda x: x
            else:
                self.relation_residual = nn.Sequential(
                    nn.Conv2d(binary_input_dim, binary_input_dim, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(binary_input_dim)
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, binary_input=None):

        res = self.residual(x)
        x, A = self.gcn(x, A) if binary_input is None else self.gcn(x, A, binary_input=binary_input)
        x = self.tcn(x) + res
        if binary_input is not None:
            n, t, v, w, c = binary_input.size()
            binary_input = binary_input.permute(0, 4, 1, 2, 3).reshape(n, c, t, v * w)
            binary_input = self.relation_residual(binary_input).permute(0, 2, 3, 1).reshape(n, -1, v, w, c)
            return self.relu(x), A, binary_input

        return self.relu(x), A


class STGCN(nn.Module):
    def __init__(self, n_agents, in_channels, n_features=256, kernel_size=(9, 7), edge_importance_weighting=True,
                 dropout=0.5, graph_option='complete', binary_input_dim=0, binary_output_dim=0, small_model=False,
                 tiny_model=False, max_gcn=False):
        """
        :param n_agents: number of agents
        :param in_channels: number of input channels for each agent
        :param n_features: length of return feature
        :param kernel_size: temporal_kernel_size and spatial_kernel_size
        :edge_importance_weighting: use edge_importance_weighting
        """
        super().__init__()

        self.n_agents = n_agents
        self.in_channels = in_channels
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.edge_importance_weighting = edge_importance_weighting
        n_graph_nodes = n_agents if graph_option == 'complete' else None
        self.graph = Graph(layout=graph_option, n_nodes=n_graph_nodes)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.A = A
        # self.register_buffer('A', A)

        assert (binary_input_dim > 0) == (binary_output_dim > 0)

        first_layer_kwargs = {'binary_input_dim': binary_input_dim,
                              'binary_output_dim': binary_output_dim,
                              'max_gcn': max_gcn}
        layer_kwargs = {'binary_input_dim': binary_output_dim,
                        'binary_output_dim': binary_output_dim,
                        'max_gcn': max_gcn}

        self.st_gcn_layers = nn.ModuleList([
            STGCN_Layer(in_channels, 64, kernel_size, 1, residual=False, **first_layer_kwargs),
            STGCN_Layer(64, 64, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(64, 64, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(64, 64, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(64, 128, kernel_size, 2, dropout=dropout, **layer_kwargs),
            STGCN_Layer(128, 128, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(128, 128, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(128, 256, kernel_size, 2, dropout=dropout, **layer_kwargs),
            STGCN_Layer(256, 256, kernel_size, 1, dropout=dropout, **layer_kwargs),
            STGCN_Layer(256, n_features, kernel_size, 1, **layer_kwargs),
        ]) if not small_model and not tiny_model else (
            nn.ModuleList([
                STGCN_Layer(in_channels, 16, kernel_size, 1, residual=False, **first_layer_kwargs),
                STGCN_Layer(16, 16, kernel_size, 1, dropout=dropout, **layer_kwargs),
                STGCN_Layer(16, 32, kernel_size, 2, dropout=dropout, **layer_kwargs),
                STGCN_Layer(32, 32, kernel_size, 1, dropout=dropout, **layer_kwargs),
                STGCN_Layer(32, 64, kernel_size, 2, dropout=dropout, **layer_kwargs),
                STGCN_Layer(64, n_features, kernel_size, 1, **layer_kwargs)
            ]) if not tiny_model else nn.ModuleList([
                STGCN_Layer(in_channels, 8, (5, 7), 1, residual=False, **first_layer_kwargs),
                STGCN_Layer(8, 12, (5, 7), 2, dropout=dropout, **layer_kwargs),
                STGCN_Layer(12, n_features, (5, 7), 1, **layer_kwargs)
            ])
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(kernel_size[1], n_agents, n_agents), requires_grad=True)
                for i in self.st_gcn_layers
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_layers)

    def forward(self, states, binary_input=None):
        """
        :param states: [batch, length, n_agents, in_channels] batched input data per frame per player.
        return [batch, n_features, length, n_agents]
        """
        device = states.device
        kernel_size = self.kernel_size
        n_features = self.n_features
        batch, length, n_agents, in_channels = states.size()
        assert n_agents == self.n_agents
        assert in_channels == self.in_channels

        # NB(Jiayuan Mao @ 04/14): add contiguous() to avoid back-propagation error in PyTorch 1.4.
        x = states.permute(0, 3, 1, 2).contiguous()

        # x[batch, in_channels, length, n_agents]
        for gcn, importance in zip(self.st_gcn_layers, self.edge_importance):
            if binary_input is not None:
                x, _, binary_input = gcn(x, self.A.to(device) * importance, binary_input=binary_input)
            else:
                x, _ = gcn(x, self.A.to(device) * importance)

        return x
