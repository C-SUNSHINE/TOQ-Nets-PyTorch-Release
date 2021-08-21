import torch


class Dependency:

    def _normalize_weights(self):
        for j, weight in enumerate(self._weights):
            s = torch.zeros(self._output_dims[j])
            for i, w in enumerate(weight):
                s += w.sum(1)
            for k in range(self._output_dims[j]):
                if s[k] < 1e-5:
                    s[k] = 1e-5
            for i, w in enumerate(weight):
                w /= s.unsqueeze(1)

    @classmethod
    def _plain_weights(cls, input_dims, output_dims):
        return [[None for in_dim in input_dims] for out_dim in output_dims]

    def __init__(self, input_dims, output_dims, weights=None, normalize=True):
        assert isinstance(input_dims, list) or isinstance(input_dims, tuple)
        assert isinstance(output_dims, list) or isinstance(output_dims, tuple)
        self._input_dims = tuple(input_dims)
        self._output_dims = tuple(output_dims)
        if weights is not None:
            self._weights = self._plain_weights(self._input_dims, self.output_dims())
            assert len(weights) == len(self._output_dims)
            for j, weight in enumerate(weights):
                assert len(weight) == len(self._input_dims)
                for i, w in enumerate(weight):
                    assert w.size() == torch.Size((self._output_dims[j], self._input_dims[i]))
                    self._weights[j][i] = torch.abs(w).clone()
        else:
            self._weights = [[torch.ones(out_dim, in_dim) for in_dim in self._input_dims] for out_dim in output_dims]
        if normalize:
            self._normalize_weights()
        self._weights = tuple([tuple(x) for x in self._weights])

    def clone(self):
        return Dependency(self._input_dims, self._output_dims, self._weights)

    def input_dims(self):
        return self._input_dims

    def output_dims(self):
        return self._output_dims

    def weights(self):
        return self._weights

    def appended(self, other, normalize=True):
        assert self.output_dims() == other.input_dims()
        mid_dims = self._output_dims
        input_dims = self.input_dims()
        output_dims = other.output_dims()
        weights = self._plain_weights(input_dims, output_dims)
        for j in range(len(output_dims)):
            for i in range(len(input_dims)):
                w = torch.zeros(output_dims[j], input_dims[i])
                for k in range(len(mid_dims)):
                    w += torch.matmul(other.weights()[j][k], self.weights()[k][i])
                weights[j][i] = w
        return Dependency(input_dims, output_dims, weights, normalize=normalize)

    def append(self, other, normalize=True):
        res = self.appended(other, normalize=normalize)
        self._input_dims = res.input_dims()
        self._output_dims = res.output_dims()
        self._weights = res.weights()
        return self

    def add(self, other, normalize=True):
        assert self.input_dims() == other.input_dims() and self.output_dims() == other.output_dims()
        w = [[self.weights()[j][i] + other.weights()[j][i] for i in range(len(self.input_dims()))
              ] for j in range(len(self.output_dims()))
             ]
        self._weights = Dependency(self.input_dims(), self.output_dims(), w, normalize=normalize).weights()

    @classmethod
    def _merge(cls, d1, d2, dim='output', normalize=False):
        if dim == 'input':
            assert d1.output_dims() == d2.output_dims()
            input_dims = d1.input_dims() + d2.input_dims()
            output_dims = d1.output_dims()
            weights = [d1.weights()[i] + d2.weights()[i] for i in range(len(output_dims))]
        elif dim == 'output':
            assert d1.input_dims() == d2.input_dims()
            input_dims = d1.input_dims()
            output_dims = d1.output_dims() + d2.output_dims()
            weights = d1.weights() + d2.weights()
        else:
            raise ValueError('dim=%s not supported' % str(dim))
        return Dependency(input_dims, output_dims, weights, normalize=normalize)

    @classmethod
    def cat(cls, ds, dim='output', normalize=False):
        res = None
        for d in ds:
            if res is None:
                res = d
            else:
                res = cls._merge(res, d, dim=dim, normalize=normalize)
        return res
