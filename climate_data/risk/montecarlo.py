"""固定种子的蒙特卡洛：基准与全部加固方案共用同一组因子抽样。

改善量按逐样本成功指示差 D_k 计算，标准误是 D 的样本标准差除以 sqrt(B)，
绝不能把两个独立概率的标准误相加（这里两份估计本就来自同一批抽样）。
抽样模式只是估计，报告中不得称为精确结果。
"""
import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Draw:
    index: int
    states: tuple[int, ...]
    partition: tuple[tuple[str, ...], ...]
    success: bool


def shared_draws(model, fortified, draws, seed):
    """生成 B 个共享因子状态抽样，并记录足以独立复演的身份信息。"""
    rng = _FixedRandom(seed)
    cumulative = []
    for factor in model.factors:
        thresholds, total = [], 0.0
        for p in factor.probs:
            total += float(p)
            thresholds.append(total)
        # 浮点累计误差归一，最后一个阈值恒为 1，保证状态总能被选中
        thresholds = [t / total for t in thresholds]
        thresholds[-1] = 1.0
        cumulative.append(thresholds)
    rows = []
    digest = hashlib.sha256()
    for k in range(draws):
        states = []
        for thresholds in cumulative:
            u = rng.random()
            chosen = 0
            while u > thresholds[chosen]:
                chosen += 1
            states.append(chosen)
        states = tuple(states)
        digest.update(bytes(states))
        rows.append(Draw(
            index=k,
            states=states,
            partition=model.terminal_partition(states, fortified),
            success=model.full_success(states, fortified),
        ))
    identity = {
        'algorithm': 'stdlib_random_Random',
        'python_rng_class': 'random.Random',
        'seed': seed,
        'draws': draws,
        'draw_method_per_factor': 'cumulative_fraction_thresholds_on_Random.random()',
        'factor_order': [f.id for f in model.factors],
        'state_order': {
            f.id: list(f.states) for f in model.factors
        },
        'matrix_sha256': digest.hexdigest(),
    }
    return rows, identity


class _FixedRandom:
    """random.Random 的薄封装；集中到一处以便复演身份描述。"""

    def __init__(self, seed):
        import random
        self._rng = random.Random(seed)

    def random(self):
        return self._rng.random()


def evaluate(model, rows, fortified):
    """在共享抽样上评估某一假设（基准或加固集合）。"""
    draws = []
    for row in rows:
        success = model.full_success(row.states, fortified)
        draws.append(Draw(row.index, row.states,
                          model.terminal_partition(row.states, fortified),
                          success))
    return draws


def paired_improvement(baseline_rows, scenario_rows):
    """逐样本配对差 D_k = I_加固,k - I_基准,k 及其样本标准误。"""
    b = len(baseline_rows)
    deltas = [
        int(s.success) - int(base.success)
        for base, s in zip(baseline_rows, scenario_rows)
    ]
    mean = sum(deltas) / b
    if b > 1:
        variance = sum((d - mean) ** 2 for d in deltas) / (b - 1)
        se = math.sqrt(variance / b)
    else:
        se = None
    return mean, se, deltas


def rate_and_se(rows):
    b = len(rows)
    rate = sum(1 for r in rows if r.success) / b
    if b > 1:
        variance = rate * (1 - rate)  # 伯努利样本方差
        se = math.sqrt(variance / b)
    else:
        se = None
    return rate, se


def mc_posterior(model, rows, fortified):
    """失败条件下的因子状态经验分布；未观察到失败则不可估计。"""
    failures = [r for r in rows if not model.full_success(r.states, fortified)]
    if not failures:
        return {'estimable': False,
                'reason': 'no_failure_observed_in_sample',
                'note': '抽样未观察到失败，不能证明真实失败概率为零'}
    counts = [[0] * len(f.states) for f in model.factors]
    for row in failures:
        for fi, state in enumerate(row.states):
            counts[fi][state] += 1
    n = len(failures)
    return {
        'estimable': True,
        'failure_draws': n,
        'states': [
            {'factor': model.factors[fi].id,
             'state_posterior': [c / n for c in counts[fi]]}
            for fi in range(len(model.factors))
        ],
    }


def representative_failures(model, rows, limit=3):
    """预览若干失败代表情景，仅用于解释；不构成对全部风险的覆盖证明。"""
    picked = []
    for row in rows:
        if not row.success:
            picked.append({
                'draw_index': row.index,
                'factor_states': {
                    model.factors[fi].id: model.factors[fi].states[state]
                    for fi, state in enumerate(row.states)
                },
                'terminal_partition': [list(g) for g in row.partition],
            })
            if len(picked) >= limit:
                break
    return picked
