"""编排精确模式与共享抽样模式，产出可独立复演的风险报告。"""
from fractions import Fraction

from ..domain import summarize
from ..io import integer
from . import exact, montecarlo
from .model import Model


def _q(value):
    """有理数序列化为最简分数字符串；未完成的空值保留 None。"""
    return None if value is None else str(value)


def _exact_payload(result):
    return {
        'method': result.method,
        'complete': result.complete,
        'termination': result.termination,
        'states_visited': result.states_visited,
        'memo_states': result.memo_states,
        'proven_probability_mass': _q(result.proven_mass),
        'connectivity_probability': _q(result.success),
        'failure_probability': _q(result.failure),
        'pinned_factor_state': (
            None if result.pinned is None
            else {'factor_index': result.pinned[0], 'state_index': result.pinned[1]}
        ),
        'unfinished_note': (
            None if result.complete
            else '预算内未完成，只报告已证明质量，不给精确概率结论'
        ),
    }


def _mc_payload(rate, se):
    return {
        'estimated_connectivity_probability': rate,
        'standard_error': se,
        'is_exact': False,
    }


def analyze(raw, digest):
    """冻结配置与情景，比较基准与每个加固方案的终端连通风险。"""
    identity = summarize(raw)
    model = Model(raw)
    state_budget = integer(raw.get('exact_state_budget', 100_000), 1)
    draws = integer(raw.get('draws', 0), 0)
    seed = raw.get('seed', 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed 应为整数')

    # 基准 = 不加固，与配置中的加固方案使用同一套机制。
    targets = [('baseline', frozenset())]
    targets += [(s.id, s.fortified) for s in model.scenarios]

    shared_rows = None
    shared_identity = None
    if draws:
        shared_rows, shared_identity = montecarlo.shared_draws(
            model, frozenset(), draws, seed
        )

    report = {
        'source_sha256': digest,
        'network_identity': identity,
        'assumptions': {
            'frozen_network': True,
            'undirected_multigraph': True,
            'independent_factors_correlated_edges': True,
            'fortification_semantics': '加固道路强制可通，其余道路机制不变',
            'success_criterion': '全部终端属于同一连通分量',
            'no_routing_or_actuation': True,
        },
        'scenarios': [],
    }

    baseline_exact = baseline_rows = None
    for sid, fortified in targets:
        exact_result = exact.solve(model, fortified, state_budget)
        posterior = (
            exact.posterior_on_failure(
                model, fortified, exact_result.failure, state_budget
            ) if exact_result.complete
            else {'defined': None, 'reason': 'exact_solver_incomplete',
                  'complete': False, 'states': None}
        )
        # 后验的 Fraction 需要统一序列化
        posterior_payload = _serialize_posterior(posterior, model)
        entry = {
            'scenario_id': sid,
            'fortified_edges': sorted(fortified),
            'exact': _exact_payload(exact_result),
            'posterior_on_failure': posterior_payload,
        }
        if shared_rows is not None:
            rows = montecarlo.evaluate(model, shared_rows, fortified)
            rate, se = montecarlo.rate_and_se(rows)
            entry['monte_carlo'] = _mc_payload(rate, se)
            entry['monte_carlo']['posterior_on_failure'] = (
                montecarlo.mc_posterior(model, rows, fortified)
            )
            entry['failure_previews'] = montecarlo.representative_failures(
                model, rows
            )
            if sid == 'baseline':
                baseline_rows = rows
            else:
                mean_delta, se_delta, deltas = montecarlo.paired_improvement(
                    baseline_rows, rows
                )
                entry['paired_improvement_vs_baseline'] = {
                    'mean_success_indicator_difference': mean_delta,
                    'paired_standard_error': se_delta,
                    'improved_draws': sum(1 for d in deltas if d > 0),
                    'degraded_draws': sum(1 for d in deltas if d < 0),
                    'unchanged_draws': sum(1 for d in deltas if d == 0),
                    'formula': 'D_k = I_加固,k - I_基准,k；SE = sd(D)/sqrt(B)',
                }
        report['scenarios'].append(entry)
        if sid == 'baseline':
            baseline_exact = exact_result

    # 精确改善量（基准与方案分别精确计算，二者同属冻结模型因此可直接相减）
    for entry in report['scenarios']:
        if entry['scenario_id'] == 'baseline':
            continue
        if baseline_exact.complete and entry['exact']['complete']:
            entry['exact_improvement_vs_baseline'] = {
                'connectivity_gain': _q(
                    _frac(entry['exact']['connectivity_probability'])
                    - baseline_exact.success
                ),
                'failure_reduction': _q(
                    baseline_exact.failure
                    - _frac(entry['exact']['failure_probability'])
                ),
            }
        else:
            entry['exact_improvement_vs_baseline'] = {
                'connectivity_gain': None,
                'failure_reduction': None,
                'note': '基准或方案未精确完成，不给精确改善结论',
            }

    # 小空间全枚举核对（只对基准做代表，方案共用同一求解器）
    report['enumeration_cross_check'] = exact.cross_check(
        model, frozenset(), state_budget
    )
    if shared_identity is not None:
        report['monte_carlo_identity'] = shared_identity
        report['monte_carlo_identity']['shared_across_all_scenarios'] = True
        report['monte_carlo_identity']['preview_notice'] = (
            '失败代表情景仅用于解释，不是全部风险的覆盖证明'
        )
    return report


def _frac(text):
    return Fraction(text) if text is not None else None


def _serialize_posterior(posterior, model):
    # 三态：可定义（完整或部分完成）、零失败概率不可定义、精确求解未完成
    if not posterior.get('defined'):
        return posterior
    factor_states = {f.id: f.states for f in model.factors}
    rows = []
    for row in posterior['states']:
        rows.append({
            'factor': row['factor'],
            'state_order': factor_states[row['factor']],
            'state_posterior': [_q(p) for p in row['state_posterior']],
        })
    return {
        'defined': True,
        'complete': posterior['complete'],
        'states': rows,
        'note': None if posterior['complete'] else '部分后验分子受预算限制未完成',
    }
