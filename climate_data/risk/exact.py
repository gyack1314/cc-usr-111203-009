"""精确有理数求解：因子赋值递归、状态合并与全枚举核对。

合并键由当前连通分量划分和每条未决道路的剩余允许状态约束共同确定。
仅“已坏边数相同”不构成等价：两条不同的失败边给出不同的残余条件，
也会改变终端之间是否仍可重连，因此绝不参与合并。
"""
import math
from dataclasses import dataclass
from fractions import Fraction

from .connectivity import DSU
from .model import PASS, FAIL, UNDECIDED


class BudgetExhausted(RuntimeError):
    """已访问状态数超出预算；已证明质量小于总空间，结论不完整。"""


@dataclass(frozen=True)
class ExactResult:
    success: Fraction | None
    failure: Fraction | None
    complete: bool
    states_visited: int
    memo_states: int
    proven_mass: Fraction
    termination: str  # complete | budget_exhausted
    method: str
    pinned: tuple[int, int] | None


def _joint_weights(model, pinned):
    """非钉住因子的先验向量；钉住位在赋值向量中固定，不参与分支。"""
    return [
        (i, f.probs) for i, f in enumerate(model.factors)
        if pinned is None or i != pinned[0]
    ]


def enumerate_full(model, fortified, pinned=None):
    """对全部因子状态做朴素加权枚举（零概率状态仍计入访问身份）。

    返回 (成功质量, 失败质量, 访问的完整状态数)。钉住因子时其余因子仍按
    各自先验加权，因此结果是给定钉住条件的条件概率（质量和为 1）。
    """
    assignment = [-1] * len(model.factors)
    if pinned is not None:
        assignment[pinned[0]] = pinned[1]
    weights = _joint_weights(model, pinned)
    totals = [Fraction(0), Fraction(0)]
    visited = 0

    def recurse(depth, weight):
        nonlocal visited
        if depth == len(weights):
            visited += 1
            totals[model.full_success(assignment, fortified)] += weight
            return
        fi, probs = weights[depth]
        for state, prob in enumerate(probs):
            assignment[fi] = state
            recurse(depth + 1, weight * prob)
            assignment[fi] = -1

    recurse(0, Fraction(1))
    return totals[1], totals[0], visited


def bounded_enumeration(model, fortified, state_budget, pinned=None):
    """字典序有界枚举：预算耗尽时如实报告已证明质量，不给概率结论。

    用递归保证每条 DFS 路径拥有自己的赋值向量视图；visited 只计完整状态。
    """
    assignment = [-1] * len(model.factors)
    if pinned is not None:
        assignment[pinned[0]] = pinned[1]
    weights = _joint_weights(model, pinned)
    total_space = math.prod(len(p) for _, p in weights)
    totals = {'success': Fraction(0), 'failure': Fraction(0),
              'proven': Fraction(0), 'visited': 0}
    complete = True

    def recurse(depth, prefix):
        nonlocal complete
        if not complete:
            return
        if depth == len(weights):
            totals['visited'] += 1
            totals['proven'] += prefix
            if model.full_success(assignment, fortified):
                totals['success'] += prefix
            else:
                totals['failure'] += prefix
            if totals['visited'] >= state_budget and totals['visited'] < total_space:
                complete = False
            return
        fi, probs = weights[depth]
        for state, prob in enumerate(probs):
            assignment[fi] = state
            recurse(depth + 1, prefix * prob)
            assignment[fi] = -1
            if not complete:
                return

    recurse(0, Fraction(1))
    return (totals['success'], totals['failure'], totals['visited'],
            totals['proven'], complete)


def _memo_solve(model, fortified, state_budget, pinned):
    """带记忆化与预算的精确递归，返回条件 (成功, 失败) 精确 Fraction。"""
    n = len(model.factors)
    assignment = [-1] * n
    if pinned is not None:
        assignment[pinned[0]] = pinned[1]
    memo = {}
    stats = {'visited': 0, 'memo': 0}
    skip = {-1} if pinned is None else {pinned[0]}

    def recurse(dsu):
        residual = []
        for edge in model.edges:
            status, cons = model.edge_status(edge, assignment, fortified)
            if status == PASS:
                dsu.union(edge.u, edge.v)
            elif status == FAIL:
                continue
            else:
                residual.append((edge, cons))
        # 终端已属同一连通分量：未决道路无关，子空间质量全部成功。
        if dsu.connected(model.terminals):
            key = ('connected', dsu.canonical(), ())
            if key not in memo:
                stats['visited'] += 1
                stats['memo'] += 1
                memo[key] = (Fraction(1), Fraction(0))
            return memo[key]
        relevant = sorted({
            fi for _, cons in residual for fi, _ in cons if fi not in skip
        })
        # 没有仍能恢复连通的道路条件：终端分离已是定局。
        if not relevant:
            stats['visited'] += 1
            return Fraction(0), Fraction(1)
        key = (dsu.canonical(), tuple(sorted(
            (edge.id, tuple(sorted(
                (fi, tuple(sorted(allowed))) for fi, allowed in cons
            )))
            for edge, cons in residual
        )))
        if key in memo:
            return memo[key]
        stats['visited'] += 1
        if stats['visited'] > state_budget:
            raise BudgetExhausted(dict(stats))
        # 约束未决道路最多的因子优先分支，尽快暴露 FAIL 与连通结局。
        counts = {fi: 0 for fi in relevant}
        for _, cons in residual:
            for fi, _ in cons:
                if fi in counts:
                    counts[fi] += 1
        fi = max(relevant, key=lambda x: (counts[x], -x))
        factor = model.factors[fi]
        agg_s = agg_f = Fraction(0)
        for state, prob in enumerate(factor.probs):
            assignment[fi] = state
            child_s, child_f = recurse(dsu.copy())
            assignment[fi] = -1
            agg_s += prob * child_s
            agg_f += prob * child_f
        value = (agg_s, agg_f)
        stats['memo'] += 1
        memo[key] = value
        return value

    result = recurse(DSU(len(model.nodes)))
    return result, stats


def solve(model, fortified, state_budget, pinned=None):
    """精确求解入口；预算耗尽时改用有界枚举报告资源与已证明质量。"""
    try:
        (success, failure), stats = _memo_solve(
            model, fortified, state_budget, pinned
        )
        return ExactResult(
            success, failure, True, stats['visited'], stats['memo'],
            success + failure, 'complete', 'memoized_factor_recursion', pinned,
        )
    except BudgetExhausted as exc:
        stats = exc.args[0]
        _, _, visited, proven, complete = bounded_enumeration(
            model, fortified, state_budget, pinned
        )
        return ExactResult(
            None, None, False, stats['visited'], stats['memo'], proven,
            'complete' if complete else 'budget_exhausted',
            'bounded_joint_enumeration', pinned,
        )


def cross_check(model, fortified, state_budget):
    """小因子空间用全枚举核对记忆化解；返回核对记录字典。"""
    if model.state_space_size > state_budget:
        return {
            'performed': False,
            'reason': 'state_space_exceeds_enumeration_budget',
            'state_space_size': model.state_space_size,
        }
    e_success, e_failure, visited = enumerate_full(model, fortified)
    memo = solve(model, fortified, state_budget)
    agree = (
        memo.complete
        and memo.success == e_success
        and memo.failure == e_failure
    )
    return {
        'performed': True,
        'agree': agree,
        'enumeration_states': visited,
        'enumeration_success': str(e_success),
        'memo_states': memo.states_visited,
        'memo_success': str(memo.success) if memo.complete else None,
    }


def posterior_on_failure(model, fortified, failure_mass, state_budget):
    """网络已失败时各因子状态的后验；失败概率为零时条件不可定义。

    P(因子=状态 | 失败) = P(因子=状态 且 失败) / P(失败)，精确有理数。
    零先验状态的联合质量为零但仍保留在配置身份中，不用先验顶替后验。
    """
    if failure_mass == 0:
        return {'defined': False, 'reason': 'failure_probability_zero',
                'complete': True, 'states': None}
    rows = []
    complete = True
    for fi, factor in enumerate(model.factors):
        posteriors = []
        for si in range(len(factor.states)):
            prior = factor.probs[si]
            if prior == 0:
                posteriors.append(Fraction(0))
                continue
            result = solve(
                model, fortified, state_budget, pinned=(fi, si)
            )
            if not result.complete:
                posteriors.append(None)
                complete = False
                continue
            joint = prior * result.failure
            posteriors.append(joint / failure_mass)
        rows.append({'factor': factor.id, 'state_posterior': posteriors})
    return {'defined': True, 'complete': complete, 'states': rows}
