"""无向多重图上的并查集与终端连通判定。"""


class DSU:
    __slots__ = ('parent', 'rank')

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def copy(self):
        clone = DSU(0)
        clone.parent = self.parent.copy()
        clone.rank = self.rank.copy()
        return clone

    def canonical(self):
        """全部节点的规范划分，用于精确求解的状态合并键。"""
        groups = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return tuple(sorted(tuple(sorted(g)) for g in groups.values()))

    def connected(self, terminals):
        first = self.find(terminals[0])
        return all(self.find(t) == first for t in terminals[1:])

    def terminal_groups(self, terminals, labels):
        """终端按连通分量分组，输出按标签排序的稳定结构。"""
        groups = {}
        for t in terminals:
            groups.setdefault(self.find(t), []).append(labels[t])
        return tuple(sorted(tuple(sorted(g)) for g in groups.values()))
