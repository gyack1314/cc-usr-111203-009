"""把冻结配置解析成因子、道路与加固情景的不可变模型。

因子之间相互独立；同一因子可约束多条道路，因此道路失效通过因子取值联动。
道路可通性是允许状态约束的合取：每个被引用因子的当前状态都须落在允许集合内。
"""
import math
from dataclasses import dataclass
from fractions import Fraction

from ..io import fraction, unique

PASS, FAIL, UNDECIDED = 1, 0, -1


@dataclass(frozen=True)
class Factor:
    id: str
    states: tuple[str, ...]
    probs: tuple[Fraction, ...]


@dataclass(frozen=True)
class Edge:
    id: str
    u: int
    v: int
    # 每项为 (因子下标, 允许状态下标集合)，合取语义；空元组表示道路始终可通
    constraints: tuple[tuple[int, frozenset[int]], ...]


@dataclass(frozen=True)
class Scenario:
    id: str
    fortified: frozenset[str]


class Model:
    """配置的索引化视图；构造前应由 domain.summarize 完成引用与概率校验。"""

    def __init__(self, raw):
        node_index = {n: i for i, n in enumerate(raw['nodes'])}
        self.nodes = tuple(raw['nodes'])
        self.terminals = tuple(node_index[t] for t in raw['terminals'])
        self.factors = tuple(
            Factor(
                f['id'],
                tuple(s['id'] for s in f['states']),
                tuple(fraction(s['probability']) for s in f['states']),
            )
            for f in raw['factors']
        )
        fpos = {f.id: i for i, f in enumerate(self.factors)}
        spos = [{s: j for j, s in enumerate(f.states)} for f in self.factors]
        edges = []
        for e in raw['edges']:
            cons = []
            for fid, allowed in e['allowed'].items():
                fi = fpos[fid]
                cons.append((fi, frozenset(spos[fi][s] for s in allowed)))
            edges.append(
                Edge(e['id'], node_index[e['u']], node_index[e['v']], tuple(sorted(cons)))
            )
        self.edges = tuple(edges)
        self.edge_pos = {e.id: i for i, e in enumerate(self.edges)}
        unique(raw['fortifications'], 'id')
        self.scenarios = tuple(
            Scenario(s['id'], frozenset(s['edges'])) for s in raw['fortifications']
        )
        self.state_space_size = math.prod(len(f.states) for f in self.factors)

    def edge_status(self, edge, assignment, fortified):
        """给定（可能部分的）因子赋值，返回道路状态与未决残约束。

        assignment 中 -1 表示该因子尚未赋值。加固道路强制可通，其余机制不变。
        返回 (PASS|FAIL|UNDECIDED, 残约束元组)；残约束只含未赋值因子。
        """
        if edge.id in fortified:
            return PASS, ()
        if not edge.constraints:
            return PASS, ()
        residual = []
        for fi, allowed in edge.constraints:
            s = assignment[fi]
            if s == -1:
                residual.append((fi, allowed))
            elif s not in allowed:
                return FAIL, ()
        if not residual:
            return PASS, ()
        return UNDECIDED, tuple(residual)

    def full_success(self, assignment, fortified):
        """完整赋值下终端是否属于同一连通分量（不只看度数）。"""
        dsu = DSU(len(self.nodes))
        for edge in self.edges:
            status, _ = self.edge_status(edge, assignment, fortified)
            if status == PASS:
                dsu.union(edge.u, edge.v)
        return dsu.connected(self.terminals)

    def terminal_partition(self, assignment, fortified):
        """完整赋值下终端的连通分量划分，返回节点身份分组。"""
        dsu = DSU(len(self.nodes))
        for edge in self.edges:
            status, _ = self.edge_status(edge, assignment, fortified)
            if status == PASS:
                dsu.union(edge.u, edge.v)
        return dsu.terminal_groups(self.terminals, self.nodes)


from .connectivity import DSU  # noqa: E402  避免循环导入的后置引用
