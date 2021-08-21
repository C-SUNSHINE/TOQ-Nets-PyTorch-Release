#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : nlm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/15/2020
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import itertools
from copy import deepcopy

import jactorch
import torch
from torch import nn

from toqnets.nn.utils import Binary
from toqnets.tools.decision_tree import get_decision_trees, get_features_from_decision_tree, predict_on_feature_set, \
    plot_decision_tree, apply_decision_trees
from toqnets.tools.predicate_dependency import Dependency
from toqnets.utils import plot_weights


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

    def __init__(self, dim, exclude_self=True, forall=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.forall = forall

    def forward(self, inputs):
        shape = inputs.size()
        inp0, inp1 = inputs, inputs
        if self.exclude_self:
            mask = exclude_mask(inputs, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(inputs, mask, 0.0)
            inp1 = mask_value(inputs, mask, 1.0)

        if self.forall:
            shape = shape[:-2] + (shape[-1] * 2,)
            exists = torch.max(inp0, dim=-2)[0]
            forall = torch.min(inp1, dim=-2)[0]
            return torch.stack((exists, forall), dim=-1).view(shape)

        shape = shape[:-2] + (shape[-1],)
        return torch.max(inp0, dim=-2)[0].view(shape)

    def get_output_dim(self, input_dim):
        if self.forall:
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
        for new_dim in self.h_dims:
            layers.append(nn.Sequential(nn.Linear(dim, new_dim), nn.ReLU()))
            dim = new_dim
        layers.append(nn.Sequential(nn.Linear(dim, self.out_dim), ))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        input_size = inputs.size()[:-1]
        input_channel = inputs.size(-1)

        return self.layers(inputs.view(-1, input_channel)).view(*input_size, -1)

    def get_dependency(self, method='weight', **kwargs):
        if method == 'weight':
            w = None
            for i in range(len(self.h_dims) + 1):
                w_new = torch.abs(self.layers[i][0].weight.detach().cpu().float())
                if w is None:
                    w = w_new
                else:
                    w = torch.mm(w_new, w)
            return Dependency([self.in_dim], [self.out_dim], [[w]])
        else:
            raise ValueError()

    def get_weights(self, normalize=False):
        w = None
        for i in range(len(self.h_dims) + 1):
            x = self.layers[i][0].weight.detach().cpu().float()
            if normalize:
                pass
                # x /= self.layers[i][0].bias.detach().cpu().float().unsqueeze(1)
            if w is None:
                w = x
            else:
                w = torch.mm(x, w)
        return w

    def get_parameter_weights(self):
        w = []
        for i in range(len(self.h_dims) + 1):
            w.append(self.layers[i][0].weight.view(-1))
        return torch.cat(w)


class MLPLogic(MLP):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__(in_dim, out_dim, h_dim)
        self.layers.add_module(str(len(self.layers)), nn.Sigmoid())


def _get_tuple_n(x, n, tp):
    """Get a length-n list of type tp."""
    assert tp is not list
    if isinstance(x, tp):
        x = [x, ] * n
    assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(
        tp)
    for i in x:
        assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
    return x


class LogicLayer(nn.Module):
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
            forall=True,
    ):
        super().__init__()
        assert breadth > 0, 'Does not support breadth <= 0.'
        assert breadth <= 3, 'Using LogicLayer with breadth > 3 may cause speed and memory issue.'

        self.max_order = breadth
        self.residual = residual
        self.permute = permute
        self.forall = forall
        self.logic_hidden_dim = deepcopy(logic_hidden_dim)
        self.data_ = {'input': [], 'output': []}
        self.trees_ = []

        input_dims = _get_tuple_n(input_dims, self.max_order + 1, int)
        output_dims = _get_tuple_n(output_dims, self.max_order + 1, int)

        self.logic, self.dim_perms, self.dim_expanders, self.dim_reducers = [
            nn.ModuleList() for _ in range(4)
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
                reducer = Reducer(i + 1, exclude_self, forall=forall)
                self.dim_reducers.append(reducer)
                current_dim += reducer.get_output_dim(input_dims[i + 1])
            else:
                self.dim_reducers.append(None)

            if current_dim == 0:
                self.dim_perms.append(None)
                self.logic.append(None)
                output_dims[i] = 0
            else:
                perm = Permutation(i, permute)
                self.dim_perms.append(perm)
                current_dim = perm.get_output_dim(current_dim)
                self.logic.append(
                    MLPLogic(current_dim, output_dims[i], self.logic_hidden_dim)
                )

        self.input_dims = deepcopy(input_dims)
        self.output_dims = deepcopy(output_dims)

        if self.residual:
            for i in range(len(input_dims)):
                self.output_dims[i] += input_dims[i]

    def forward(self, inputs, binary, keep_data=False, use_trees=False):
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
                f = self.dim_perms[i](f)
                if use_trees:
                    output = apply_decision_trees(self.trees_[i], f)
                else:
                    output = self.logic[i](f)
                if binary:
                    output = self.binary(output)
            if self.residual and self.input_dims[i] > 0:
                output = torch.cat([inputs[i], output], dim=-1)
            outputs.append(output)
        if keep_data:
            self.data_['output'].append([(x.detach().cpu() if x is not None else None) for x in outputs])
        return outputs

    def get_edges(self, pname, qname):
        edges = []
        for i in range(self.max_order + 1):
            out_cnt = 0
            interm = []
            if i > 0 and self.input_dims[i - 1] > 0:
                out_cnt += self.dim_expanders[i].get_output_dim(self.input_dims[i - 1])
                for k in range(self.input_dims[i - 1]):
                    interm.append((i - 1, k, 'expand'))
                assert (out_cnt == len(interm))
            if i < len(self.input_dims) and self.input_dims[i] > 0:
                out_cnt += self.input_dims[i]
                for k in range(self.input_dims[i]):
                    interm.append((i, k, ''))
                assert (out_cnt == len(interm))
            if i + 1 < len(self.input_dims) and self.input_dims[i + 1] > 0:
                out_cnt += self.dim_reducers[i].get_output_dim(self.input_dims[i + 1])
                for k in range(self.input_dims[i + 1]):
                    interm.append((i + 1, k, 'exists'))
                    interm.append((i + 1, k, 'forall'))
                assert (out_cnt == len(interm))

            if out_cnt != 0:
                assert not self.permute
                assert len(interm) == self.logic[i].in_dim
                residual_offset = self.input_dims[i] if self.residual else 0
                mlp_weight = self.logic[i].get_weights(normalize=True)
                assert self.logic[i].out_dim == self.output_dims[i] - residual_offset
                for j in range(self.logic[i].out_dim):
                    # print('%s_%d_%d' % (qname, i, j + residual_offset))
                    for pp, (ii, kk, e_label) in enumerate(interm):
                        edges.append((
                            '%s_%d_%d' % (pname, ii, kk),
                            '%s_%d_%d' % (qname, i, j + residual_offset),
                            e_label,
                            float(mlp_weight[j, pp])
                        ))
                if self.residual:
                    for j in range(residual_offset):
                        # print('%s_%d_%d' % (qname, i, j))
                        edges.append((
                            '%s_%d_%d' % (pname, i, j),
                            '%s_%d_%d' % (qname, i, j),
                            '=',
                            float(1)
                        ))

        return edges

    def get_parameter_weights(self):
        return torch.cat([mlp.get_parameter_weights() for mlp in self.logic])

    def _get_data(self):
        inputs = [None for i in range(self.max_order + 1)]
        outputs = [None for i in range(self.max_order + 1)]
        for x in self.data_['input']:
            for i in range(self.max_order + 1):
                inputs[i] = merge(inputs[i], x[i], dim=0)
        for x in self.data_['output']:
            for i in range(self.max_order + 1):
                outputs[i] = merge(outputs[i], x[i], dim=0)
        return inputs, outputs

    def show(self, last_rep, rep, log=print, save_dir=''):
        vname = rep[0][0][0]
        inputs, outputs = self._get_data()
        binary_net = Binary()
        inputs = [binary_net(x) for x in inputs]
        outputs = [binary_net(x) for x in outputs]
        for i in range(self.max_order + 1):
            out_cnt = 0
            interm = []
            f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                n = inputs[i].size(1) if i == 1 else None
                f.append(self.dim_expanders[i](inputs[i - 1], n))
                out_cnt += f[-1].size(-1)
                for k in range(self.input_dims[i - 1]):
                    interm.append('forall_x %s(..., x)' % last_rep[i - 1][k])
                assert (out_cnt == len(interm))
            if i < len(inputs) and self.input_dims[i] > 0:
                f.append(inputs[i])
                out_cnt += f[-1].size(-1)
                for k in range(self.input_dims[i]):
                    interm.append('%s(..., x)' % last_rep[i][k])
                assert (out_cnt == len(interm))
            if i + 1 < len(inputs) and self.input_dims[i + 1] > 0:
                f.append(self.dim_reducers[i](inputs[i + 1]))
                out_cnt += f[-1].size(-1)
                for k in range(self.input_dims[i + 1]):
                    interm.append('exists_x %s(..., x)' % last_rep[i + 1][k])
                    interm.append('forall_x %s(..., x)' % last_rep[i + 1][k])
                assert (out_cnt == len(interm))

            if self.residual and self.input_dims[i] > 0:
                for k in range(self.input_dims[i]):
                    log('%s = %s' % (rep[i][k], last_rep[i][k]))
            if out_cnt == 0:
                output = None
            else:
                f = torch.cat(f, dim=-1)
                # f = self.dim_perms[i](f)
                assert not self.permute
                assert len(interm) == self.logic[i].in_dim
                assert len(interm) == f[i].size(-1)
                residual_offset = self.input_dims[i] if self.residual else 0
                assert outputs[i].size(-1) == self.logic[i].out_dim + residual_offset

                mlp_output = outputs[i].view(-1, outputs[i].size(-1))[:, residual_offset:]
                mlp_output = mlp_output.view(*outputs[i].size()[:-1], self.logic[i].out_dim)
                assert f.size(-1) == self.logic[i].in_dim
                trees, trees_inputs_mean, trees_outputs_mean = get_decision_trees(f, mlp_output)
                in_names = interm
                out_names = rep[i][residual_offset:]
                logic_weights = self.logic[i].get_weights()
                plot_weights(logic_weights, in_names, out_names, log=log, save_dir=save_dir)
                plot_decision_tree(save_dir, trees, in_names, out_names,
                                   log=log, inputs_mean=trees_inputs_mean, outputs_mean=trees_outputs_mean)
                self.trees_.append(trees)
                assert len(out_names) == self.logic[i].out_dim
                tree_features = [get_features_from_decision_tree(dt, size_threshold=0.05) for dt in trees]
                # print(tree_features)
                approx_tree_outputs = torch.stack(
                    [predict_on_feature_set(dt, fs, f) for dt, fs in zip(trees, tree_features)],
                    dim=-1
                )
                # formula = get_logic_formula(f, approx_tree_outputs, in_names, tree_features)
                formula = ['decision_tree(%s)' % x for x in out_names]
                for k, out_name in enumerate(out_names):
                    log('%s = %s' % (out_name, formula[k]))

    def get_dependency(self, method='weight', log=print, save_dir=''):
        weights = [[torch.zeros(out_dim, in_dim) for in_dim in self.input_dims] for out_dim in self.output_dims]
        use_data = method in ['decision_tree']
        if use_data:
            inputs, outputs = self._get_data()
            binary_net = Binary()
            inputs = [binary_net(x) for x in inputs]
            outputs = [binary_net(x) for x in outputs]
        else:
            inputs, outputs = None, None

        for i in range(self.max_order + 1):
            out_cnt = 0
            interm = []
            if use_data:
                f = []
            if i > 0 and self.input_dims[i - 1] > 0:
                if use_data:
                    n = inputs[i].size(1) if i == 1 else None
                    f.append(self.dim_expanders[i](inputs[i - 1], n))
                out_cnt += self.dim_expanders[i].get_output_dim(self.input_dims[i - 1])
                for k in range(self.input_dims[i - 1]):
                    interm.append((i - 1, k))
                assert (out_cnt == len(interm))
            if i < len(self.input_dims) and self.input_dims[i] > 0:
                if use_data:
                    f.append(inputs[i])
                out_cnt += self.input_dims[i]
                for k in range(self.input_dims[i]):
                    interm.append((i, k))
                assert (out_cnt == len(interm))
            if i + 1 < len(self.input_dims) and self.input_dims[i + 1] > 0:
                if use_data:
                    f.append(self.dim_reducers[i](inputs[i + 1]))
                out_cnt += self.dim_reducers[i].get_output_dim(self.input_dims[i + 1])
                for k in range(self.input_dims[i + 1]):
                    interm.append((i + 1, k))
                    if self.forall:
                        interm.append((i + 1, k))
                assert (out_cnt == len(interm))

            if self.residual and self.input_dims[i] > 0:
                weights[i][i][:self.input_dims[i], :] = torch.eye(self.input_dims[i])
            if out_cnt != 0:
                if use_data:
                    f = torch.cat(f, dim=-1)
                assert not self.permute
                assert len(interm) == self.logic[i].in_dim
                residual_offset = self.input_dims[i] if self.residual else 0
                if method == 'weight':
                    mlp_dependency = self.logic[i].get_dependency(method='weight', log=log, save_dir=save_dir)
                    assert self.logic[i].out_dim == self.output_dims[i] - residual_offset
                    for pp, (ii, kk) in enumerate(interm):
                        weights[i][ii][residual_offset:, kk] = mlp_dependency.weights()[0][0][:, pp]
                elif method == 'decision_tree':
                    mlp_output = outputs[i].view(-1, outputs[i].size(-1))[:, residual_offset:]
                    mlp_output = mlp_output.view(*outputs[i].size()[:-1], self.logic[i].out_dim)
                    assert f.size(-1) == self.logic[i].in_dim
                    trees, trees_inputs_mean, trees_outputs_mean = get_decision_trees(f, mlp_output, max_depth=4)
                    tree_features = [get_features_from_decision_tree(dt, size_threshold=0.05) for dt in trees]
                    for j, fs in enumerate(tree_features):
                        for f in fs:
                            ii, kk = interm[f]
                            weights[i][ii][residual_offset + j, kk] = 1
                else:
                    raise ValueError("Invalid dependency computation method")
        return Dependency(self.input_dims, self.output_dims, weights)


class LogicMachine(nn.Module):
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
            forall=True,
    ):
        super().__init__()
        self.depth = depth
        self.breadth = breadth
        self.residual = residual
        self.io_residual = io_residual
        self.recursion = recursion
        self.connections = connections
        self.input_dims = tuple(input_dims)

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
            layer = LogicLayer(breadth, current_dims, output_dims, logic_hidden_dim,
                               exclude_self, residual, permute=permute, forall=forall)
            current_dims = deepcopy(layer.output_dims)
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

    def forward(self, inputs, depth=None, binary_layer=False, keep_data=False, use_trees=False):
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

            f = layer(f, binary=binary_layer, keep_data=keep_data, use_trees=use_trees)
            f = self._mask(f, i, None)
            if self.io_residual:
                for j, out in enumerate(f):
                    outputs[j] = merge(outputs[j], out)
        if not self.io_residual:
            outputs = f
        return outputs

    def get_parameter_weights(self):
        return torch.cat([layer.get_parameter_weights() for layer in self.layers])

    def get_nodes(self, pnames):
        nodes = []
        for i in range(self.depth):
            dims = self.layers[i].output_dims
            for j in range(len(dims)):
                for k in range(dims[j]):
                    nodes.append('%s_%d_%d' % (pnames[i], j, k))
        return nodes

    def get_edges(self, ppname, pnames):
        edges = []
        for i in range(self.depth):
            edges += self.layers[i].get_edges((ppname if i == 0 else pnames[i - 1]), pnames[i])
        return edges

    def show(self, input_dims, log=print, save_dir=''):
        current_dims = deepcopy(input_dims)
        rep = [['P_%d_%d' % (j, k) for k in range(current_dims[j])] for j in range(len(current_dims))]
        # log(rep)
        # log(current_dims)
        v_name = ['Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        assert self.depth <= len(v_name)
        for i in range(self.depth):
            current_dims = deepcopy(self.layers[i].output_dims)
            new_rep = [['%s_%d_%d' % (v_name[i], j, k) for k in range(current_dims[j])] for j in
                       range(len(current_dims))]
            self.layers[i].show(rep, new_rep, log=log, save_dir=save_dir)
            # log(current_dims)
            rep = new_rep
        return rep

    def get_dependency(self, input_dims, method='weight', log=print, save_dir=''):
        """
        :param input_dims: the input_dims of the inputed predicates, nullary, unary, binary, and so on
        :param method: the method of compute attentions
        return dependency on each layer, dependency between layer
        """
        dependencies = []
        layer_dependencies = []
        primitive_dependencies = []
        current_dims = input_dims
        for i in range(self.depth):
            last_dims = current_dims
            current_dims = deepcopy(self.layers[i].output_dims)
            layer_dependency = self.layers[i].get_dependency(method=method, log=log, save_dir=save_dir)
            print(layer_dependency.input_dims(), layer_dependency.output_dims())
            # print(last_dims, current_dims)
            assert layer_dependency.input_dims() == tuple(last_dims) and layer_dependency.output_dims() == tuple(
                current_dims)
            for x in dependencies:
                x.append(layer_dependency)
            dependencies.append(layer_dependency.clone())
            layer_dependencies.append(layer_dependency.clone())
        return dependencies, layer_dependencies
