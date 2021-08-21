import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import tree


def get_decision_trees(inputs, outputs, max_depth=3):
    inputs = inputs.view(-1, inputs.size(-1))
    outputs = outputs.view(-1, outputs.size(-1))
    trees = []
    for k in range(outputs.size(-1)):
        X = inputs.numpy()
        Y = outputs[:, k].numpy()
        clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(X, Y)
        trees.append(clf)
    return trees, inputs.float().mean(dim=0), outputs.float().mean(dim=0)


def apply_decision_trees(trees, inputs):
    device = inputs.device
    inputs = inputs.detach().cpu()
    size = inputs.size()
    inputs = inputs.view(-1, inputs.size(-1)).numpy()
    outputs = np.zeros(inputs.shape[:-1] + (len(trees),))
    for k in range(len(trees)):
        clf = trees[k]
        outputs[:, k] = clf.predict(inputs)
    outputs = torch.from_numpy(outputs).float().view(*size[:-1], len(trees)).to(device)
    return outputs


def plot_decision_tree(save_dir, trees, in_name, out_name, log=print, inputs_mean=None, outputs_mean=None):
    os.makedirs(save_dir, exist_ok=True)
    assert len(out_name) == len(trees)
    for k, clf in enumerate(trees):
        plt.figure(figsize=(50, 50))
        tree.plot_tree(clf)
        plt.savefig(os.path.join(save_dir, out_name[k] + '.png'))
        plt.close()
        txt = open(os.path.join(save_dir, out_name[k] + '.txt'), 'w')
        for j, s in enumerate(in_name):
            if inputs_mean is not None:
                txt.write('X[%2d] = %s, average = %.2lf\n' % (j, s, float(inputs_mean[j])))
            else:
                txt.write('X[%2d] = %s\n' % (j, s))
        txt.close()


def get_features_from_decision_tree(dt, split_threshold=0.1, size_threshold=0.05):
    """
    Given a decision tree, return a set of variables which we need to consider only on the decision tree to get approximate predictions
    """
    t = dt.tree_
    lc = t.children_left
    rc = t.children_right
    feature = t.feature
    size = t.n_node_samples
    min_size = size[0] * size_threshold
    f = set()
    q = [0]
    qh = 0
    while qh < len(q):
        x = q[qh]
        qh += 1
        if lc[x] != -1:
            if feature[x] not in f:
                cur_size = size[x]
                cur_split = min(size[lc[x]], size[rc[x]]) / cur_size
                if cur_size >= min_size and cur_split >= split_threshold:
                    f.add(feature[x])
                    q.append(lc[x])
                    q.append(rc[x])
            else:
                q.append(lc[x])
                q.append(rc[x])
    return f


def predict_on_feature_set(dt, f, inputs):
    """
    Predict on decision tree t, considering only feature set f, on inputs, whose last dimension are the vatiables.
    inputs should be torch tensor
    """
    t = dt.tree_
    labels = dt.classes_
    size = inputs.size()
    inputs = inputs.view(-1, size[-1])

    lc = t.children_left
    rc = t.children_right
    feature = t.feature
    threshold = t.threshold
    value = t.value

    q = [0]
    qh = 0
    mask = [None for i in range(t.node_count)]
    mask[0] = torch.ones(inputs.size(0), dtype=torch.bool)
    outputs = torch.zeros(inputs.size(0), dtype=torch.bool)
    while qh < len(q):
        x = q[qh]
        qh += 1
        if lc[x] != -1 and feature[x] in f:
            left = torch.le(inputs[:, feature[x]], threshold[x]) & mask[x]
            right = (~left) & mask[x]
            mask[lc[x]] = left
            mask[rc[x]] = right
            q.append(lc[x])
            q.append(rc[x])
        else:
            maxc = None
            label = -1
            for k in range(labels.shape[0]):
                if maxc is None or value[x, 0, k] > maxc:
                    label = labels[k]
                    maxc = value[x, 0, k]
            outputs[mask[x]] = True if float(label) > .5 else False
    outputs = outputs.view(size[:-1])
    return outputs
