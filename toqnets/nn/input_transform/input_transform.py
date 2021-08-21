#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : input_transform.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

import jactorch
import torch
import torch.nn as nn
from jacinle.utils.enum import JacEnum

from toqnets.nn.input_transform.nn_based import NullaryPrimitivesNN, UnaryPrimitivesNN, BinaryPrimitivesNN, \
    TrinaryPrimitiveNN
from toqnets.nn.input_transform.predefined import NullaryPrimitivesPredefined, UnaryPrimitivesPredefined, \
    BinaryPrimitivesPredefined, NullaryPrimitivesPredefined_v2, UnaryPrimitivesPredefined_v2, \
    BinaryPrimitivesPredefined_v2
from toqnets.nn.input_transform.primitives import MaxPoolTrinary, MinPoolTrinary
from toqnets.nn.input_transform.something_else import NullaryPrimitivesSomethingElse, UnaryPrimitivesSomethingElse, \
    BinaryPrimitivesSomethingElse
from toqnets.nn.input_transform.toyota import NullaryPrimitivesToyotaJoint, UnaryPrimitivesToyotaJoint, \
    BinaryPrimitivesToyotaJoint, TrinaryPtimitivesToyotaJoint, NullaryPrimitivesToyotaSkeleton, \
    UnaryPrimitivesToyotaSkeleton

__all__ = [
    'TimeReductionMethod', 'InputTransformTrinaryTime',
    'InputTransformPredefined', 'InputTransformPredefined_v2', 'InputTransformNN',
    'InputTransformToyotaJoint', 'InputTransformToyotaSkeleton',
    'InputTransformSomethingElse',
    'InputTransformVidvrd'  # Not implemented.
]


class TimeReductionMethod(JacEnum):
    TERNARY_MIN_MAX = 'ternary_min_max'
    NONE = 'none'


class InputTransformTrinaryTime(nn.Module):
    def __init__(self, add_nullary_dim=None, add_unary_dim=None, n_arys=None, time_reduction='ternary_min_max',
                 **kwargs):
        super().__init__()

        self.order = len(n_arys)
        self.add_nullary_dim = add_nullary_dim
        self.add_unary_dim = add_unary_dim if 'type_dim' not in kwargs else kwargs['type_dim']
        if 'type_dim' in kwargs:
            kwargs.pop('type_dim')
        self.n_ary_primitives = nn.ModuleList()
        if n_arys is not None:
            for n, n_ary in enumerate(n_arys):
                self.n_ary_primitives.add_module(str(n), n_ary(**kwargs))
        self.time_reduction = TimeReductionMethod.from_string(time_reduction)

        dim_multiplier = 1
        if self.time_reduction is TimeReductionMethod.TERNARY_MIN_MAX:
            self.minpool = MinPoolTrinary()
            self.maxpool = MaxPoolTrinary()
            dim_multiplier = 6
        else:
            pass

        self.out_dims = [
            self.n_ary_primitives[i].out_dim * dim_multiplier for i in range(len(self.n_ary_primitives))
        ]
        if self.add_nullary_dim is not None:
            self.out_dims[0] += self.add_nullary_dim
        if self.add_unary_dim is not None:
            self.out_dims[1] += self.add_unary_dim

    def forward_time_reduction_ternary_min_max(self, x):
        x = torch.cat([self.minpool(x), self.maxpool(x)], dim=1)
        sizes = list(x.size())
        new_sizes = sizes[:1] + sizes[2:-1] + [sizes[1] * sizes[-1]]
        return x.permute(*([i for i in range(len(sizes)) if i != 1] + [1])).reshape(*new_sizes)

    def forward_time_reduction(self, x):
        if self.time_reduction is TimeReductionMethod.TERNARY_MIN_MAX:
            return self.forward_time_reduction_ternary_min_max(x)
        elif self.time_reduction is TimeReductionMethod.NONE:
            return x
        else:
            raise ValueError('Unknown time reduction method: {}.'.format(self.time_reduction))

    def expand_to_nullary_shape(self, nullary_add_tensor, nullary_tensor):
        if self.time_reduction is TimeReductionMethod.TERNARY_MIN_MAX:
            return nullary_add_tensor
        elif self.time_reduction is TimeReductionMethod.NONE:
            assert nullary_tensor is not None
            return jactorch.add_dim(nullary_add_tensor, 1, nullary_tensor.size(1))  # [batch_size, time, hidden]
        else:
            raise ValueError('Unknown time reduction method: {}.'.format(self.time_reduction))

    def expand_to_unary_shape(self, unary_add_tensor, unary_tensor):
        if self.time_reduction is TimeReductionMethod.TERNARY_MIN_MAX:
            return unary_add_tensor
        elif self.time_reduction is TimeReductionMethod.NONE:
            assert unary_tensor is not None
            return jactorch.add_dim(unary_add_tensor, 1, unary_tensor.size(1))  # [batch_size, time, agent, hidden]
        else:
            raise ValueError('Unknown time reduction method: {}.'.format(self.time_reduction))

    def forward(self, states, special_nullary_states=None, add_nullary_tensor=None, add_unary_tensor=None, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        """
        results = []
        for i in range(len(self.n_ary_primitives)):
            if i == 0 and special_nullary_states is not None:
                results.append(self.n_ary_primitives[i](special_nullary_states, beta=beta))
            else:
                results.append(self.n_ary_primitives[i](states, beta=beta))

        results = [self.forward_time_reduction(x) if x is not None else None for x in results]

        if add_nullary_tensor is not None:
            add_nullary_tensor = self.expand_to_nullary_shape(add_nullary_tensor, results[0])
            if results[0] is None:
                results[0] = add_nullary_tensor.type(torch.float)
            else:
                results[0] = torch.cat([results[0], add_nullary_tensor.type(torch.float)], dim=-1)

        if add_unary_tensor is not None:
            add_unary_tensor = self.expand_to_unary_shape(add_unary_tensor, results[1])
            if results[1] is None:
                results[1] = add_unary_tensor.type(torch.float)
            else:
                results[1] = torch.cat([results[1], add_unary_tensor.type(torch.float)], dim=-1)

        return results

    def estimate_parameters(self, states):
        for i in range(len(self.n_ary_primitives)):
            self.n_ary_primitives[i](states, estimate_parameters=True)

    def reset_parameters(self, parameter_name):
        for i in range(len(self.n_ary_primitives)):
            self.n_ary_primitives[i].reset_parameters(parameter_name)

    def require_grad(self, mode):
        for param in self.parameters():
            param.requires_grad_(mode)

    def show(self, log=print, save_dir=None):
        for order, primitive in enumerate(self.n_ary_primitives):
            if self.time_reduction == TimeReductionMethod.from_string('ternary_min_max'):
                descriptions = primitive.get_descriptions()
                cnt = 0
                for k in range(len(descriptions)):
                    for i, (quantize, stage) in enumerate([('forall', 'pre'), ('forall', 'act'), ('forall', 'eff'),
                                                           ('exists', 'pre'), ('exists', 'act'), ('exists', 'eff')]):
                        log("P_%d_%d = %s_%s %s" % (order, cnt, quantize, stage, descriptions[k]))
                        cnt += 1
                if order == 1 and self.add_unary_dim is not None:
                    for k in range(self.add_unary_dim):
                        log("P_%d_%d = type_%d" % (order, cnt, k))
                        cnt += 1
            elif self.time_reduction == TimeReductionMethod.from_string('none'):
                descriptions = primitive.get_descriptions()
                cnt = 0
                for k in range(len(descriptions)):
                    log("P_%d_%d = %s" % (order, cnt, descriptions[k]))
                    cnt += 1
                if order == 1 and self.add_unary_dim is not None:
                    for k in range(self.add_unary_dim):
                        log("P_%d_%d = type_%d" % (order, cnt, k))
                        cnt += 1
            else:
                raise NotImplementedError()

    def get_nodes(self, prefix='P'):
        nodes = {}
        for order, primitive in enumerate(self.n_ary_primitives):
            assert self.time_reduction == TimeReductionMethod.NONE
            descriptions = primitive.get_descriptions()
            for i in range(len(descriptions)):
                nodes['%s_%d_%d' % (prefix, order, i)] = descriptions[i]
        return nodes


class InputTransformPredefined(InputTransformTrinaryTime):
    def __init__(self, **kwargs):
        super().__init__(
            n_arys=[NullaryPrimitivesPredefined, UnaryPrimitivesPredefined, BinaryPrimitivesPredefined],
            **kwargs
        )


class InputTransformPredefined_v2(InputTransformTrinaryTime):
    def __init__(self, **kwargs):
        super().__init__(
            n_arys=[NullaryPrimitivesPredefined_v2, UnaryPrimitivesPredefined_v2, BinaryPrimitivesPredefined_v2],
            **kwargs
        )


class InputTransformNN(InputTransformTrinaryTime):
    def __init__(self, breadth=2, **kwargs):
        if breadth == 1:
            super().__init__(
                n_arys=[NullaryPrimitivesNN, UnaryPrimitivesNN],
                **kwargs
            )
        elif breadth == 2:
            super().__init__(
                n_arys=[NullaryPrimitivesNN, UnaryPrimitivesNN, BinaryPrimitivesNN],
                **kwargs
            )
        elif breadth == 3:
            super().__init__(
                n_arys=[NullaryPrimitivesNN, UnaryPrimitivesNN, BinaryPrimitivesNN, TrinaryPrimitiveNN],
                **kwargs
            )
        else:
            raise ValueError()


class InputTransformVidvrd(nn.Module):
    def __init__(self, state_dim, type_dim, ):
        super(InputTransformVidvrd, self).__init__()

    def forward(self, states, types, beta):
        raise NotImplementedError()


class InputTransformToyotaJoint(InputTransformTrinaryTime):
    def __init__(self, **kwargs):
        super().__init__(n_arys=[
            NullaryPrimitivesToyotaJoint,
            UnaryPrimitivesToyotaJoint,
            BinaryPrimitivesToyotaJoint,
            TrinaryPtimitivesToyotaJoint
        ], **kwargs)


class InputTransformToyotaSkeleton(InputTransformTrinaryTime):
    def __init__(self, **kwargs):
        super().__init__(n_arys=[
            NullaryPrimitivesToyotaSkeleton,
            UnaryPrimitivesToyotaSkeleton,
        ], **kwargs)


class InputTransformSomethingElse(InputTransformTrinaryTime):
    def __init__(self, **kwargs):
        super().__init__(n_arys=[
            NullaryPrimitivesSomethingElse,
            UnaryPrimitivesSomethingElse,
            BinaryPrimitivesSomethingElse
        ], **kwargs)


class InputTransformTCN(InputTransformTrinaryTime):
    def __init__(self, breadth=2, **kwargs):
        if breadth == 1:
            super().__init__(
                n_arys=[NullaryPrimitivesTCN, UnaryPrimitivesTCN],
                **kwargs
            )
        elif breadth == 2:
            super().__init__(
                n_arys=[NullaryPrimitivesTCN, UnaryPrimitivesTCN, BinaryPrimitivesTCN],
                **kwargs
            )
        elif breadth == 3:
            super().__init__(
                n_arys=[NullaryPrimitivesTCN, UnaryPrimitivesTCN, BinaryPrimitivesTCN, TrinaryPrimitiveTCN],
                **kwargs
            )
        else:
            raise ValueError()

    def forward(self, states, special_nullary_states=None, add_nullary_tensor=None, add_unary_tensor=None, beta=None):
        """
        :param states: [batch, length, n_agents, state_dim]
        """
        results = super().forward(states, special_nullary_states=None, add_nullary_tensor=None,
                                  add_unary_tensor=None, beta=None)
        new_length = results[1].size(1)

        if add_nullary_tensor is not None:
            add_nullary_tensor = add_nullary_tensor[:, :new_length]
            add_nullary_tensor = self.expand_to_nullary_shape(add_nullary_tensor, results[0])
            if results[0] is None:
                results[0] = add_nullary_tensor.type(torch.float)
            else:
                results[0] = torch.cat([results[0], add_nullary_tensor.type(torch.float)], dim=-1)

        if add_unary_tensor is not None:
            add_unary_tensor = add_unary_tensor[:, :new_length]
            add_unary_tensor = self.expand_to_unary_shape(add_unary_tensor, results[1])
            if results[1] is None:
                results[1] = add_unary_tensor.type(torch.float)
            else:
                results[1] = torch.cat([results[1], add_unary_tensor.type(torch.float)], dim=-1)

        return results

