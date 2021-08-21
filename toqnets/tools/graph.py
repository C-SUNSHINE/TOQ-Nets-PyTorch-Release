import os

from graphviz import Digraph


class MyGraph:
    """
    edge format (start, end, label, weight(with sign))
    """

    def __init__(self, nodes=None, edges=None):
        if nodes is None:
            assert edges is not None
            self.nodes = tuple(set([x[0] for x in edges] + [x[1] for x in edges]))
            self.edges = tuple(edges)
        else:
            self.nodes = tuple(nodes)
            self.edges = tuple(edges) if edges is not None else tuple()

    def filter_inward_edges(self, top=3, except_nodes=None):
        new_edges = []
        for v in self.nodes:
            ine = [x for x in self.edges if x[1] == v]
            ine = list(reversed((sorted(ine, key=lambda x: abs(x[3])))))
            if except_nodes is None or v not in except_nodes:
                ine = ine[:top]
            new_edges.extend(ine)
        return MyGraph(self.nodes, new_edges)

    def get_component(self, start, reverse=False):
        q = [start]
        qe = []
        vis = {start}
        adj = {k: [e for e in self.edges if e[1 if reverse else 0] == k] for k in self.nodes}
        i = 0
        while i < len(q):
            x = q[i]
            i += 1
            for e in adj[x]:
                y = e[0 if reverse else 1]
                if y not in vis:
                    vis.add(y)
                    q.append(y)
                qe.append(e)
        return MyGraph(q, qe)

    def dfs(self, x, node_label, sign):
        if x in node_label:
            if sign == -1:
                if '>' in node_label[x]:
                    return node_label[x].replace('>', '<')
                elif '<' in node_label[x]:
                    return node_label[x].replace('<', '>')
                else:
                    return node_label[x].replace('=', '!=')
            else:
                return node_label[x]
        adje = [e for e in self.edges if e[1] == x]
        adje = list(sorted(adje, key=lambda x: (1 if 'C' in x[0] else 0)))
        elems = []
        inv = {
            'exists': 'forall',
            'forall': 'exists',
            'until': 'then',
            'then': 'until',
        }
        print(x, adje)
        for y, _, tp, w in adje:
            new_sign = -sign if w < 0 else sign
            if new_sign == -1 and tp in inv:
                new_tp = inv[tp]
            else:
                new_tp = tp
            if new_tp in inv:
                elems.append(new_tp + '(%s)' % self.dfs(y, node_label, new_sign))
            else:
                elems.append(self.dfs(y, node_label, new_sign))
        return ' & '.join(elems)

    def show(self, node_label):
        os.makedirs('dumps/interpretability', exist_ok=True)
        fout = open('dumps/interpretability/graph.txt', 'w')
        fout.write(repr(self.nodes) + ',')
        fout.write(repr(self.edges) + ',')
        fout.write(repr(node_label) + '\n')

        deg = {k: 0 for k in self.nodes}
        anno = {v: (node_label[v] if v in node_label else None) for v in self.nodes}
        for e in self.edges:
            deg[e[1]] += 1
        q = []
        for v in deg.keys():
            if deg[v] == 0:
                q.append(v)
        i = 0
        while i < len(q):
            x = q[i]
            i += 1
            adje = [e for e in self.edges if e[0] == x]
            for e in adje:
                assert x == e[0]
                y = e[1]
                print(x, y, anno[x])
                deg[y] -= 1
                assert deg[y] >= 0

                sign = '-' if e[3] < 0 else ''
                annox = '-(%s)' % anno[x] if e[3] < 0 else anno[x]
                if anno[y] == None:
                    if e[2] in ['exists', 'forall']:
                        anno[y] = e[2] + '(' + annox + ')'
                    elif e[2] in ['then', 'until']:
                        anno[y] = e[2] + '(' + annox + ')'
                    else:
                        anno[y] = annox
                else:
                    if e[2] in ['exists', 'forall']:
                        anno[y] += ' & ' + e[2] + '(' + annox + ')'
                    elif e[2] in ['then', 'until']:
                        if 'C' in x:
                            anno[y] += ' & ' + e[2] + '(' + annox + ')'
                        else:
                            anno[y] = e[2] + '(' + annox + ')' + ' & ' + anno[y]
                    else:
                        anno[y] += ' & ' + annox
                if deg[y] == 0:
                    q.append(y)
        fout.write('\n' * 10)
        for k in q:
            fout.write(k + '  : ' + self.dfs(k, node_label, 1) + '\n')
        fout.close()

        dot = Digraph(comment='interpretability NLTL')
        for v in self.nodes:
            dot.node(v, label=node_label[v] if v in node_label else None)
        for e in self.edges:
            dot.edge(e[0], e[1], '%.2f %s' % (e[3], e[2]))
        dot.render('dumps/interpretability/graph.gv', view=True)
