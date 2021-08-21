#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : nltl.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/15/2020
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import itertools

import jactorch
import torch
from torch import nn

from toqnets.nn.utils import Binary
from .modules import TemporalPooling1D, TemporalPooling2D

__all__ = ['TemporalLogicLayer', 'TemporalLogicMachine']


def merge(x, y, dim=-1):
    if x is None:
        return y
    if y is None:
        return x
    return torch.cat([x, y], dim=dim)


def exclude_mask(inputs, cnt=2, dim=1):
    """Produce an exclusive mask.
    Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).
    Args:
      inputs: The tensor to be masked.
      cnt: The operation is performed over [dim, dim + cnt) axes.
      dim: The starting dimension for the exclusive mask.
    Returns:
      A mask that make sure the coordinates are mutually exclusive.
    """
    assert cnt > 0
    if dim < 0:
        dim += inputs.dim()
    n = inputs.size(dim)
    for i in range(1, cnt):
        assert n == inputs.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    # Mutually Exclusive
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(inputs.dim() - dim - cnt):
        mask.unsqueeze_(-1)

    return mask.expand(inputs.size()).float()


def mask_value(inputs, mask, value):
    assert inputs.size() == mask.size()
    return inputs * mask + value * (1 - mask)


class Compose(nn.ModuleList):
    def get_output_dim(self, input_dim):
        for module in self.children():
            input_dim = module.get_output_dim(input_dim)
        return input_dim


class Expander(nn.Module):
    """Capture a free variable into predicates, implemented by broadcast."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs, n=None):
        # print(inputs.size(), self.dim)
        if self.dim == 0:
            assert n is not None
        elif n is None:
            n = inputs.size(-2)
        return jactorch.add_dim(inputs, -2, n)
        # return inputs.unsqueeze(self.dim + 1).repeat(*([1] * (self.dim + 1) + [n, 1]))

    def get_output_dim(self, input_dim):
        return input_dim


class Reducer(nn.Module):
    """Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling."""

    def __init__(self, dim, exclude_self=True, exists=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.exists = exists

    def forward(self, inputs):
        shape = inputs.size()
        inp0, inp1 = inputs, inputs
        if self.exclude_self:
            mask = exclude_mask(inputs, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(inputs, mask, 0.0)
            inp1 = mask_value(inputs, mask, 1.0)

        if self.exists:
            shape = shape[:-2] + (shape[-1] * 2,)
            exists = torch.max(inp0, dim=-2)[0]
            forall = torch.min(inp1, dim=-2)[0]
            return torch.stack((exists, forall), dim=-1).view(shape)

        shape = shape[:-2] + (shape[-1],)
        return torch.max(inp0, dim=-2)[0].view(shape)

    def get_output_dim(self, input_dim):
        if self.exists:
            return input_dim * 2
        return input_dim


class Permutation(nn.Module):
    """Create r! new predicates by permuting the axies for r-arity predicates."""

    def __init__(self, dim, permute=True):
        super().__init__()
        self.dim = dim
        self.permute = permute

    def forward(self, inputs):
        if self.dim <= 1 or not self.permute:
            return inputs
        nr_dims = len(inputs.size())
        # Assume the last dim is channel.
        index = tuple(range(nr_dims - 1))
        start_dim = nr_dims - 1 - self.dim
        assert start_dim > 0
        res = []
        for i in itertools.permutations(index[start_dim:]):
            p = index[:start_dim] + i + (nr_dims - 1,)
            res.append(inputs.permute(p))
        return torch.cat(res, dim=-1)

    def get_output_dim(self, input_dim):
        if not self.permute:
            return input_dim
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dims):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dims = tuple(h_dims)
        layers = []
        dim = in_dim
        for new_dim in self.h_dims + (out_dim,):
            layers.append(nn.Linear(dim, new_dim))
            dim = new_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        input_size = inputs.size()[:-1]
        input_channel = inputs.size(-1)

        return self.layers(inputs.view(-1, input_channel)).view(*input_size, -1)


class MLPLogic(MLP):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__(in_dim, out_dim, h_dim)
        self.layers.add_module(str(len(self.layers)), nn.Sigmoid())


def _get_tuple_n(x, n, tp):
    """Get a length-n list of type tp."""
    assert tp is not list
    if isinstance(x, tp):
        x = tuple([x, ] * n)
    assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(tp)
    for i in x:
        assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
    return tuple(x)


class TemporalLogicLayer(nn.Module):
    """Logic Layers do one-step differentiable logic deduction.
    The predicates grouped by their number of variables. The inter group deduction
    is done by expansion/reduction, the intra group deduction is done by logic
    model.
    Args:
      breadth: The breadth of the logic layer.
      input_dims: the number of input channels of each input group, should consist
                  with the inputs. use dims=0 and input=None to indicate no input
                  of that group.
      output_dims: the number of output channels of each group, could
                   use a single value.
      logic_hidden_dim: The hidden dim of the logic model.
      exclude_self: Not allow multiple occurrence of same variable when
                    being True.
      residual: Use residual connections when being True.
    """

    def __init__(
            self,
            breadth,
            input_dims,
            output_dims,
            logic_hidden_dim,
            exclude_self=True,
            residual=False,
            permute=True,
            temporal=True,
            temporal_logic_dimension=1
    ):
        super().__init__()
        assert breadth > 0, 'Does not support breadth <= 0.'
        assert breadth <= 3, 'Using TemporalLogicLayer with breadth > 3 may cause speed and memory issue.'

        self.max_order = breadth
        self.residual = residual
        self.permute = permute
        self.temporal = temporal
        self.temporal_logic_dimension = temporal_logic_dimension

        self.logic_hidden_dim = tuple(logic_hidden_dim)
        self.data_ = {'input': [], 'output': []}

        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
        output_dims = list(_get_tuple_n(output_dims, self.max_order + 1, int))

        self.logic, self.dim_perms, self.dim_temporal_pools, self.dim_expanders, self.dim_reducers = [
            nn.ModuleList() for _ in range(5)
        ]
        self.binary = Binary()
        for i in range(self.max_order + 1):
            # collect current_dim from group i-1, i and i+1.
            current_dim = input_dims[i]
            if i > 0:
                expander = Expander(i - 1)
                self.dim_expanders.append(expander)
                current_dim += expander.get_output_dim(input_dims[i - 1])
            else:
                self.dim_expanders.append(None)

            if i + 1 < self.max_order + 1:
                reducer = Reducer(i + 1, exclude_self)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0:
                self.dim_temporal_pools.append(None)
                self.dim_perms.append(None)
                self.logic.append(None)
                output_dims[i] = 0
            else:
                if self.temporal_logic_dimension == 1:
                    temporal_pool = TemporalPooling1D()
                elif self.temporal_logic_dimension == 2:
                    temporal_pool = TemporalPooling2D()
                else:
                    raise ValueError('Invalid temporal logic: {}.'.format(self.temporal_logic_dimension))

                self.dim_temporal_pools.append(temporal_pool)
                current_dim = temporal_pool.get_output_dim(current_dim)

                perm = Permutation(i, permute)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)

                logic = MLPLogic(current_dim, output_dims[i], self.logic_hidden_dim)
                self.logic.append(logic)

        if self.residual:
            for i in range(len(input_dims)):
                output_dims[i] += input_dims[i]

        self.input_dims = tuple(input_dims)
        self.output_dims = tuple(output_dims)

    def forward(self, inputs, binary, keep_data=False):
        assert len(inputs) == self.max_order + 1
        if binary:
            inputs = [(self.binary(x) if x is not None else None) for x in inputs]
        outputs = []

        if keep_data:
            self.data_['input'].append([(x.detach().cpu() if x is not None else None) for x in inputs])

        for i in range(self.max_order + 1):
            # collect input f from group i-1, i and i+1.
            f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                n = inputs[i].size(-2) if i == 1 else None
                f.append(self.dim_expanders[i](inputs[i - 1], n))
            if i < len(inputs) and self.input_dims[i] > 0:
                f.append(inputs[i])
            if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                f.append(self.dim_reducers[i](inputs[i + 1]))
            if len(f) == 0:
                output = None
            else:
                f = torch.cat(f, dim=-1)
                f = self.dim_temporal_pools[i](f)
                f = self.dim_perms[i](f)
                output = self.logic[i](f)
                if binary:
                    output = self.binary(output)
            if self.residual and self.input_dims[i] > 0:
                output = torch.cat([inputs[i], output], dim=-1)
            outputs.append(output)

        if keep_data:
            self.data_['output'].append([(x.detach().cpu() if x is not None else None) for x in outputs])

        return outputs


class TemporalLogicMachine(nn.Module):
    """Neural Logic Machine consists of multiple logic layers."""

    def __init__(
            self,
            depth,
            breadth,
            input_dims,
            output_dims,
            logic_hidden_dim,
            exclude_self=True,
            residual=False,
            io_residual=False,
            recursion=False,
            connections=None,
            permute=True,
            temporal_logic_dimension=1
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.recursion = recursion
        self.connections = connections
        self.temporal_logic_dimension = temporal_logic_dimension

        assert not (self.residual and self.io_residual), \
            'Only one type of residual connection is allowed at the same time.'

        # element-wise addition for vector
        def add_(x, y):
            for i in range(len(y)):
                x[i] += y[i]
            return x

        self.layers = nn.ModuleList()
        current_dims = input_dims
        total_output_dims = [0 for _ in range(self.breadth + 1)
                             ]  # for IO residual only
        for i in range(depth):
            # IO residual is unused.
            if i > 0 and io_residual:
                add_(current_dims, input_dims)
            # Not support output_dims as list or list[list] yet.
            layer = TemporalLogicLayer(
                breadth, current_dims, output_dims, logic_hidden_dim,
                exclude_self, residual, permute=permute, temporal_logic_dimension=temporal_logic_dimension
            )
            current_dims = tuple(layer.output_dims)
            current_dims = self._mask(current_dims, i, 0)
            if io_residual:
                add_(total_output_dims, current_dims)
            self.layers.append(layer)

        if io_residual:
            self.output_dims = total_output_dims
        else:
            self.output_dims = current_dims
        # for d in range(depth):
        #     print('%d' % d, self.layers[d].output_dims)

    # Mask out the specific group-entry in layer i, specified by self.connections.
    # For debug usage.
    def _mask(self, a, i, masked_value):
        if self.connections is not None:
            assert i < len(self.connections)
            mask = self.connections[i]
            if mask is not None:
                assert len(mask) == len(a)
                a = [x if y else masked_value for x, y in zip(a, mask)]
        return a

    def forward(self, inputs, depth=None, binary_layer=False, keep_data=False):
        outputs = [None for _ in range(self.breadth + 1)]
        f = inputs

        # depth: the actual depth used for inference
        if depth is None:
            depth = self.depth
        if not self.recursion:
            depth = min(depth, self.depth)

        layer = None
        last_layer = None
        for i in range(depth):
            if i > 0 and self.io_residual:
                for j, inp in enumerate(inputs):
                    f[j] = merge(f[j], inp)
            # To enable recursion, use scroll variables layer/last_layer
            # For weight sharing of period 2, i.e. 0,1,2,1,2,1,2,...
            if self.recursion and i >= 3:
                assert not self.residual
                layer, last_layer = last_layer, layer
            else:
                last_layer = layer
                layer = self.layers[i]

            f = layer(f, binary=binary_layer, keep_data=keep_data)
            f = self._mask(f, i, None)
            if self.io_residual:
                for j, out in enumerate(f):
                    outputs[j] = merge(outputs[j], out)
        if depth == 0:
            f[:, :, :] = f.max(dim=1, keepdim=True)
            outputs = f
        if not self.io_residual:
            outputs = f
        return outputs
