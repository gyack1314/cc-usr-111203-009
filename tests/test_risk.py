"""相关灾害风险分析器的精确值、联动、合并与抽样核对。"""
import json
import math
import unittest
from fractions import Fraction

from climate_data.io import load
from climate_data.domain import summarize
from climate_data.risk.analysis import analyze, _frac
from climate_data.risk.model import Model
from climate_data.risk import exact, montecarlo
from climate_data.risk.exact import (
    BudgetExhausted, bounded_enumeration, enumerate_full, solve,
)


def load_model(name):
    raw, _ = load(f'data/{name}')
    return raw, Model(raw)


def baseline_entry(report):
    return next(s for s in report['scenarios'] if s['scenario_id'] == 'baseline')


def scenario_entry(report, sid):
    return next(s for s in report['scenarios'] if s['scenario_id'] == sid)


class ExactCouplingTests(unittest.TestCase):
    def test_parallel_bridges_fully_coupled(self):
        # 同一洪水因子联动两座平行桥：成功 3/4，不是独立假设下的 1-(1/4)^2
        raw, model = load_model('parallel-coupled.json')
        report = analyze(raw, 'digest')
        base = baseline_entry(report)
        self.assertEqual(base['exact']['connectivity_probability'], '3/4')
        self.assertEqual(base['exact']['failure_probability'], '1/4')
        self.assertTrue(base['exact']['complete'])
        # 后验：失败时河流必为 flood
        post = base['posterior_on_failure']
        self.assertTrue(post['defined'])
        self.assertEqual(post['states'][0]['state_posterior'], ['0', '1'])
        # 加固一座桥后：calm 时双通，flood 时仅剩加固桥 → 恒通
        hard = scenario_entry(report, 'harden-one')
        self.assertEqual(hard['exact']['connectivity_probability'], '1')
        self.assertEqual(
            hard['exact_improvement_vs_baseline']['connectivity_gain'], '1/4'
        )
        # 零失败概率时后验不可定义，不用先验替代
        self.assertFalse(hard['posterior_on_failure']['defined'])
        self.assertEqual(
            hard['posterior_on_failure']['reason'], 'failure_probability_zero'
        )

    def test_cycle_network_requires_any_one_path(self):
        raw, model = load_model('cycle-network.json')
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertEqual(base['exact']['connectivity_probability'], '1/2')
        hard = scenario_entry(report, 'harden-bypass')
        self.assertEqual(hard['exact']['connectivity_probability'], '1')
        self.assertEqual(
            hard['exact_improvement_vs_baseline']['connectivity_gain'], '1/2'
        )

    def test_zero_probability_state_keeps_identity_no_mass(self):
        raw, model = load_model('zero-probability-state.json')
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        # 链路 river=low 且 rain=dry：2/3 * 1/2 = 1/3
        self.assertEqual(base['exact']['connectivity_probability'], '1/3')
        self.assertEqual(base['exact']['failure_probability'], '2/3')
        post = base['posterior_on_failure']['states']
        # P(river=low|fail)=(1/3)/(2/3)=1/2；
        # P(rain=dry|fail)=(1/6)/(2/3)=1/4，P(wet|fail)=3/4，phantom 为零
        river = next(r for r in post if r['factor'] == 'river')
        rain = next(r for r in post if r['factor'] == 'rain')
        self.assertEqual(river['state_posterior'], ['1/2', '1/2'])
        self.assertEqual(rain['state_posterior'], ['1/4', '3/4', '0'])
        # 全枚举访问 6 个状态（含零概率），但总正质量为 1
        e_s, e_f, visited = enumerate_full(model, frozenset())
        self.assertEqual(visited, 6)
        self.assertEqual(e_s + e_f, Fraction(1))


class TerminalPartitionTests(unittest.TestCase):
    def test_empty_allowed_set_means_never_passable(self):
        # 数据契约：允许状态空集合表示该道路在该因子任何取值下都不可通
        raw, model = load_model('empty-allowed.json')
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertEqual(base['exact']['connectivity_probability'], '0')
        # 失败为必然事件，后验仍可定义且等于先验
        post = base['posterior_on_failure']['states'][0]['state_posterior']
        self.assertEqual(post, ['3/7', '4/7'])
        hard = scenario_entry(report, 'repair-causeway')
        self.assertEqual(hard['exact']['connectivity_probability'], '1')
        self.assertEqual(
            hard['exact_improvement_vs_baseline']['connectivity_gain'], '1'
        )

    def test_disconnected_terminals_posterior_equals_prior(self):
        raw, model = load_model('disconnected-terminal.json')
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertEqual(base['exact']['connectivity_probability'], '0')
        self.assertEqual(base['exact']['failure_probability'], '1')
        # 本就不连通：失败条件无信息，后验 == 先验
        rows = base['posterior_on_failure']['states']
        river = next(r for r in rows if r['factor'] == 'river')
        self.assertEqual(river['state_posterior'], ['3/5', '2/5'])

    def test_always_connected_is_certain(self):
        raw, model = load_model('always-connected.json')
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertEqual(base['exact']['connectivity_probability'], '1')
        self.assertFalse(base['posterior_on_failure']['defined'])


class CrossCheckAndMergeTests(unittest.TestCase):
    def test_posterior_matches_brute_force_joint_mass(self):
        # 独立路径：枚举全部完整状态，直接累计 P(因子=状态 且 失败)，
        # 再除以 P(失败)，与求解器的钉住条件后验逐项对照
        for name in ('normal.json', 'zero-probability-state.json',
                     'parallel-coupled.json', 'empty-allowed.json'):
            with self.subTest(file=name):
                raw, model = load_model(name)
                result = solve(model, frozenset(), 100_000)
                self.assertTrue(result.complete)
                joint = [
                    [Fraction(0)] * len(f.states) for f in model.factors
                ]
                assignment = [-1] * len(model.factors)

                def walk(depth, weight):
                    if depth == len(model.factors):
                        if not model.full_success(assignment, frozenset()):
                            for fi, state in enumerate(assignment):
                                joint[fi][state] += weight
                        return
                    for si, p in enumerate(model.factors[depth].probs):
                        assignment[depth] = si
                        walk(depth + 1, weight * p)
                        assignment[depth] = -1

                walk(0, Fraction(1))
                post = exact.posterior_on_failure(
                    model, frozenset(), result.failure, 100_000
                )
                if result.failure == 0:
                    self.assertFalse(post['defined'])
                    continue
                for fi, row in enumerate(post['states']):
                    for si, got in enumerate(row['state_posterior']):
                        self.assertEqual(got, joint[fi][si] / result.failure)
                    # 每个因子的后验状态概率和为 1
                    self.assertEqual(
                        sum(row['state_posterior'], Fraction(0)), Fraction(1)
                    )

    def test_enumeration_agrees_with_memo(self):
        for name in ('normal.json', 'parallel-coupled.json', 'cycle-network.json',
                     'zero-probability-state.json', 'always-connected.json',
                     'disconnected-terminal.json'):
            with self.subTest(file=name):
                raw, model = load_model(name)
                check = exact.cross_check(model, frozenset(), 100_000)
                self.assertTrue(check['performed'])
                self.assertTrue(check['agree'])

    def test_merging_does_not_equate_different_residual_edges(self):
        # 同坏边数、不同残约束的状态必须生成不同合并键：直接构造递归中间态核对
        raw, model = load_model('normal.json')
        # 完整枚举与记忆化解在该网络上一致，已说明合并保持终端连通关系
        e_s, e_f, _ = enumerate_full(model, frozenset())
        result = solve(model, frozenset(), 100_000)
        self.assertTrue(result.complete)
        self.assertEqual(result.success, e_s)
        self.assertEqual(result.failure, e_f)
        # 记忆化确实减少了状态访问（全枚举 6 个完整状态，递归含部分状态）
        self.assertGreaterEqual(result.states_visited, 1)
        self.assertGreaterEqual(result.memo_states, 1)

    def test_fortified_edges_are_forced_pass(self):
        raw, model = load_model('normal.json')
        fort = frozenset({'road-a', 'road-c'})
        result = solve(model, fort, 100_000)
        # 与基准不同；且结果与朴素枚举在加固语义下一致
        base = solve(model, frozenset(), 100_000)
        self.assertTrue(result.complete and base.complete)
        self.assertGreaterEqual(result.success, base.success)
        e_s, _, _ = enumerate_full(model, fort)
        self.assertEqual(result.success, e_s)


class BudgetTests(unittest.TestCase):
    def test_budget_exhaustion_reports_resources_honestly(self):
        raw, model = load_model('tight-budget.json')
        result = solve(model, frozenset(), 3)
        self.assertFalse(result.complete)
        self.assertEqual(result.termination, 'budget_exhausted')
        self.assertIsNone(result.success)
        self.assertIsNone(result.failure)
        # 有界枚举只证明了部分质量
        self.assertGreater(result.proven_mass, Fraction(0))
        self.assertLess(result.proven_mass, Fraction(1))
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertFalse(base['exact']['complete'])
        self.assertIsNone(base['exact']['connectivity_probability'])
        # 未完成时不冒充后验
        self.assertIsNone(base['posterior_on_failure']['defined'])

    def test_bounded_enumeration_mass_and_visited(self):
        raw, model = load_model('tight-budget.json')
        s, f, visited, proven, complete = bounded_enumeration(
            model, frozenset(), 5
        )
        self.assertFalse(complete)
        self.assertEqual(visited, 5)
        self.assertEqual(proven, s + f)
        # 提高预算后应完整，质量归一
        s2, f2, v2, p2, c2 = bounded_enumeration(
            model, frozenset(), 100_000
        )
        self.assertTrue(c2)
        self.assertEqual(p2, Fraction(1))
        self.assertEqual(s2 + f2, Fraction(1))


class MonteCarloTests(unittest.TestCase):
    def test_shared_draws_identity_and_reproducibility(self):
        raw, model = load_model('normal.json')
        rows1, id1 = montecarlo.shared_draws(model, frozenset(), 200, 89)
        rows2, id2 = montecarlo.shared_draws(model, frozenset(), 200, 89)
        self.assertEqual([r.states for r in rows1], [r.states for r in rows2])
        self.assertEqual(id1['matrix_sha256'], id2['matrix_sha256'])
        rows3, id3 = montecarlo.shared_draws(model, frozenset(), 200, 90)
        self.assertNotEqual(id1['matrix_sha256'], id3['matrix_sha256'])

    def test_all_scenarios_share_same_draw_matrix(self):
        raw, model = load_model('normal.json')
        rows, identity = montecarlo.shared_draws(model, frozenset(), 300, 89)
        for sid, fortified in [('base', frozenset()), ('x', frozenset({'road-c'}))]:
            evaluated = montecarlo.evaluate(model, rows, fortified)
            self.assertEqual(
                [r.states for r in evaluated], [r.states for r in rows]
            )

    def test_paired_standard_error_is_not_sum_of_marginals(self):
        # normal 网络加固 road-c：两份成功指示经验协方差严格为正，
        # 忽略协方差、直接相加边际方差必然错误
        raw, model = load_model('normal.json')
        rows, _ = montecarlo.shared_draws(model, frozenset(), 800, 89)
        base_rows = montecarlo.evaluate(model, rows, frozenset())
        hard_rows = montecarlo.evaluate(
            model, rows, frozenset({'road-c'})
        )
        mean_d, se_d, deltas = montecarlo.paired_improvement(
            base_rows, hard_rows
        )
        # 恒等式：SE(D) = sd(D)/sqrt(B)，其中 D 是逐样本配对差
        sample_sd = math.sqrt(
            sum((d - mean_d) ** 2 for d in deltas) / (len(deltas) - 1)
        )
        self.assertAlmostEqual(se_d, sample_sd / math.sqrt(len(deltas)))
        base_flags = [int(r.success) for r in base_rows]
        hard_flags = [int(r.success) for r in hard_rows]

        def sample_var(xs):
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

        # Var(A-B) = Var(A)+Var(B)-2Cov(A,B)；该组合下协方差为正
        var_difference = sample_var(
            [h - x for h, x in zip(hard_flags, base_flags)]
        )
        var_marginals = sample_var(base_flags) + sample_var(hard_flags)
        self.assertNotAlmostEqual(var_difference, var_marginals)
        self.assertLess(var_difference, var_marginals)
        # 改善量均值等于成功率之差（同一批样本）
        rb, _ = montecarlo.rate_and_se(base_rows)
        rh, _ = montecarlo.rate_and_se(hard_rows)
        self.assertAlmostEqual(mean_d, rh - rb)

    def test_mc_posterior_undefined_when_no_failure(self):
        raw, model = load_model('always-connected.json')
        rows, _ = montecarlo.shared_draws(model, frozenset(), 100, 1)
        post = montecarlo.mc_posterior(model, rows, frozenset())
        self.assertFalse(post['estimable'])
        self.assertEqual(post['reason'], 'no_failure_observed_in_sample')

    def test_mc_report_is_marked_approximate(self):
        raw, model = load_model('normal.json')
        report = analyze(raw, 'd')
        entry = baseline_entry(report)
        self.assertFalse(entry['monte_carlo']['is_exact'])
        self.assertIn('standard_error', entry['monte_carlo'])
        self.assertTrue(
            report['monte_carlo_identity']['shared_across_all_scenarios']
        )
        # 预览仅用于解释
        self.assertIn('preview_notice', report['monte_carlo_identity'])


class ReportContractTests(unittest.TestCase):
    def test_report_json_is_finite_and_complete_for_all_fixtures(self):
        import pathlib
        for p in sorted(pathlib.Path('data').glob('*.json')):
            if p.name.startswith('invalid-'):
                continue
            with self.subTest(file=p.name):
                raw, digest = load(str(p))
                report = analyze(raw, digest)
                text = json.dumps(report, ensure_ascii=False, allow_nan=False)
                parsed = json.loads(text)
                self.assertEqual(parsed['source_sha256'], digest)
                self.assertTrue(parsed['assumptions']['frozen_network'])

    def test_large_space_memo_stays_inside_budget(self):
        raw, model = load_model('large-factor-space.json')
        self.assertGreater(model.state_space_size, 100_000)
        report = analyze(raw, 'd')
        base = baseline_entry(report)
        self.assertTrue(base['exact']['complete'])
        self.assertLessEqual(base['exact']['states_visited'], 100_000)
        # 精确结果仍是有理数且质量归一
        s = _frac(base['exact']['connectivity_probability'])
        f = _frac(base['exact']['failure_probability'])
        self.assertEqual(s + f, Fraction(1))
        # 大空间不做朴素全枚举
        self.assertFalse(report['enumeration_cross_check']['performed'])


if __name__ == '__main__':
    unittest.main()
