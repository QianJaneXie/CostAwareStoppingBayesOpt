"""Exercise empirical notebook rules and cache loops on distinct toy trajectories.

Only selected analysis cells execute, with synthetic inputs and temporary outputs;
no experiment caches are read or changed. Run with unittest discovery in tests/.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import dill
import numpy as np

from test_lcbench_full_stopping import RULE_NAMES, trajectory

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    'LCBench_KnownCost', 'LCBench_UnknownCost',
    'LCBench_NATS_KnownCost', 'NATS_SSS_KnownCost',
)
RAW_FILES = {
    'LCBench_KnownCost': 'lcbench_known_cost_metrics_per_acq_six_datasets.pkl',
    'LCBench_UnknownCost': 'lcbench_unknown_cost_metrics_per_acq.pkl',
    'NATS_SSS_KnownCost': 'nats_sss_known_cost_metrics_per_acq_updated.pkl',
}


def cells(name):
    notebook = json.loads((ROOT / 'notebooks' / f'{name}.ipynb').read_text())
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def acquisitions(name):
    if name == 'LCBench_UnknownCost':
        return ['PBGI(1e-4)', 'LogEIPC-exp', 'LogEIPC-inv', 'LCB', 'TS']
    return ['PBGI(1e-4)', 'LogEIPC', 'LCB', 'TS']


def load_rules(name, runs, dataset='toy'):
    namespace = {'metrics_per_acq': {dataset: runs},
                 'best_acc_per_dataset': {dataset: 100.}, 'dill': dill,
                 'duplicate_seed_cleanup': []}
    source = next(s for s in cells(name) if 'templates = [' in s)
    exec(compile(source, name, 'exec'), namespace)
    rules = {r['stp_key']: r['condition_fn'] for r in namespace['stopping_rules']
             if r['acq_key'] == 'PBGI(1e-4)' and r['stp_key'] in RULE_NAMES}
    return namespace, rules


class EmpiricalStoppingCombinationTests(unittest.TestCase):
    def test_every_rule_requires_and_uses_the_selected_trajectory(self):
        for name in NOTEBOOKS:
            expected = dict(zip(acquisitions(name), [22, 30, 26, 34, 28]))
            runs = {a: trajectory(stop) for a, stop in expected.items()}
            for run in runs.values():
                run['LogEIPC-inv acq'] = run['LogEIPC acq'].copy()
            namespace, rules = load_rules(name, runs)
            self.assertEqual(set(rules), set(RULE_NAMES))
            for rule_name, condition in rules.items():
                with self.subTest(notebook=name, rule=rule_name):
                    with self.assertRaises(TypeError):
                        condition(22, 0, 'toy')
                    for acquisition, stop in expected.items():
                        actual = next(i for i in range(20, 40)
                                      if condition(i, 0, 'toy', fa=acquisition))
                        self.assertEqual(actual, stop, acquisition)

    def test_metric_and_median_belong_to_the_selected_trajectory(self):
        for name in NOTEBOOKS:
            # The combined notebook has a different SRGap threshold for NATS.
            datasets = ['toy', 'cifar10-valid'] if name == 'LCBench_NATS_KnownCost' else ['toy']
            for dataset in datasets:
                with self.subTest(notebook=name, dataset=dataset):
                    selected = acquisitions(name)[1]
                    runs = {a: trajectory(30) for a in acquisitions(name)}
                    for run in runs.values():
                        run['LogEIPC-inv acq'] = run['LogEIPC acq'].copy()
                    namespace, rules = load_rules(name, runs, dataset)
                    run = runs[selected]
                    run['current best observed'][0, 20:22] = [10., 9.]
                    run['PBGI(1e-4) acq'][0, 21] = 9.5
                    # Current incumbent crosses; previous incumbent and LogEI do not.
                    self.assertTrue(rules['PBGI'](21, 0, dataset, fa=selected))
                    self.assertFalse(rules['PBGI'](21, 0, dataset, fa='PBGI(1e-4)'))
                    metric = 'LogEIPC-inv acq' if 'Unknown' in name else 'LogEIPC acq'
                    run[metric][0, 1:21] = 10.
                    run[metric][0, 21] = 4.
                    chi = .01 if name == 'NATS_SSS_KnownCost' or dataset == 'cifar10-valid' else .1
                    run['exp min regret gap'][0, 1:21] = 100.
                    run['exp min regret gap'][0, 21] = 50. * chi
                    for rule_name in ('LogEIPC-med', 'SRGap-med'):
                        self.assertTrue(rules[rule_name](21, 0, dataset, fa=selected))
                        self.assertFalse(rules[rule_name](21, 0, dataset, fa='PBGI(1e-4)'))

    def test_immediate_standard_error_counts_each_seed_once(self):
        # Execute the collectors and statistics from the actual plotting cells.
        # Repeating identical initial trajectories under more acquisition labels
        # must not create extra independent trials or shrink the error band.
        checked = 0
        for name in ('NATS_SSS_KnownCost', 'LCBench_NATS_KnownCost'):
            for source in cells(name):
                for parent in ast.walk(ast.parse(source)):
                    body = getattr(parent, 'body', None)
                    if not isinstance(body, list):
                        continue
                    for position, statement in enumerate(body):
                        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                            continue
                        target = statement.targets[0]
                        if not isinstance(target, ast.Name) or target.id not in ('immediate_vals', 'immediate_normalized_vals'):
                            continue
                        value_name = target.id
                        if position + 1 == len(body) or not isinstance(body[position + 1], ast.For):
                            continue
                        branch = next((node for node in body[position + 2:]
                                       if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                                       and node.test.id == value_name), None)
                        if branch is None:
                            continue
                        stats = [node for node in branch.body if isinstance(node, ast.Assign)]
                        program = compile(ast.Module(body=[statement, body[position + 1], *stats], type_ignores=[]), name, 'exec')
                        normalized = value_name == 'immediate_normalized_vals'
                        for count in (2, 4):
                            with self.subTest(notebook=name, normalized=normalized, acquisitions=count):
                                order = ['PBGI', 'LogEIPC', 'LCB', 'TS'][:count]
                                initial = np.arange(50, dtype=float)
                                runs = {}
                                for index, a in enumerate(order):
                                    permutation = np.roll(np.arange(50), 7 * index)
                                    key = 'PBGI(1e-4)' if a == 'PBGI' else a
                                    runs[key] = {
                                        'config id': permutation[:, None],
                                        'final test error': initial[permutation, None],
                                        'estimated cumulative cost': np.zeros((50, 1)),
                                        'cumulative cost': np.zeros((50, 1)),
                                    }
                                regrets = {a: {r: dict(enumerate(values)) for r, values in {
                                    'Immediate': initial, 'PBGI': np.full(50, 50.),
                                    'Hindsight': np.zeros(50),
                                }.items()} for a in order}
                                namespace = {'np': np, 'd': 'toy', 'lam': 1e-4,
                                             'lam_str': '1e-4', 'best_error': 0.,
                                             'acq_order': order, 'lcbench_datasets': ['toy'],
                                             'metrics_per_acq': {'toy': runs},
                                             'cost_key': 'cumulative cost',
                                             'cost_adjusted_regrets': {'1e-4': {'toy': regrets}},
                                             'stopping_rules': [{'acq_key': 'PBGI(1e-4)', 'stp_key': r}
                                                                for r in ('Immediate', 'PBGI', 'Hindsight')]}
                                exec(program, namespace)
                                expected = initial / 50. if normalized else initial
                                self.assertAlmostEqual(namespace['immediate_mean'], expected.mean())
                                self.assertAlmostEqual(namespace['immediate_err'], expected.std() / np.sqrt(50))
                        checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_combined_loader_removes_only_the_verified_duplicate_run(self):
        reference = np.column_stack([np.arange(50), np.arange(50) + 100])
        reference[3, 0] = reference[2, 0]  # Same start can be legitimate.
        duplicated = np.insert(reference, 1, reference[0], axis=0)
        metrics = {'APSFailure': {
            'PBGI(1e-3)': {'config id': duplicated,
                           'current best observed': np.arange(51)[:, None]},
            'LogEIPC': {'config id': reference},
        }}
        source = cells('LCBench_NATS_KnownCost')[0]
        with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
            folder = Path('empirical_results')
            folder.mkdir()
            raw = folder / 'lcbench_known_cost_metrics_per_acq_all_datasets.pkl'
            with raw.open('wb') as stream:
                dill.dump(metrics, stream)
            with (folder / RAW_FILES['NATS_SSS_KnownCost']).open('wb') as stream:
                dill.dump({}, stream)
            namespace = {}
            exec(compile(source, 'combined loader', 'exec'), namespace)
            result = namespace['lcbench_metrics_per_acq']['APSFailure']['PBGI(1e-3)']
            np.testing.assert_array_equal(result['config id'], reference)
            np.testing.assert_array_equal(np.asarray(result['current best observed'])[:, 0],
                                          [0, *range(2, 51)])
            with raw.open('rb') as stream:
                self.assertEqual(len(dill.load(stream)['APSFailure']['PBGI(1e-3)']['config id']), 51)
            duplicated[1, 1] = -1  # An unexpected run must not be silently dropped.
            with raw.open('wb') as stream:
                dill.dump(metrics, stream)
            with self.assertRaises(ValueError):
                exec(compile(source, 'combined loader', 'exec'), {})

    def test_cache_loops_use_each_pair_warmup_and_cap(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name), tempfile.TemporaryDirectory() as temporary:
                # A pre-warmup crossing, a different crossing for each acquisition,
                # and a crossing beyond the paper cap expose dispatch/boundary bugs.
                length = 401 if name in ('NATS_SSS_KnownCost', 'LCBench_NATS_KnownCost') else 201
                rounds = np.arange(length)
                expected, runs = {}, {}
                for acquisition, middle in zip(acquisitions(name), [170, 180, 190, 199, 175]):
                    crossings = np.array([5, middle, 300])[:, None]
                    shape = (3, length)
                    runs[acquisition] = {
                        'current best observed': np.zeros(shape),
                        'PBGI(1e-4) acq': np.where(rounds < crossings, -1., 1.),
                        'LogEIPC acq': np.zeros(shape),
                        'LogEIPC-inv acq': np.zeros(shape),
                        'exp min regret gap': np.ones(shape),
                        'regret upper bound': np.ones(shape),
                        'PRB': np.zeros(shape),
                        'final test error': np.tile(rounds, (3, 1)),
                        'estimated cumulative cost': np.zeros(shape),
                        'cumulative cost': np.zeros(shape),
                    }
                    expected[acquisition] = [20, middle, 200]
                namespace, _ = load_rules(name, runs)
                with contextlib.chdir(temporary), contextlib.redirect_stdout(io.StringIO()):
                    Path('empirical_results').mkdir()
                    if name in RAW_FILES:
                        with (Path('empirical_results') / RAW_FILES[name]).open('wb') as stream:
                            dill.dump({'toy': runs}, stream)
                    if name == 'NATS_SSS_KnownCost':
                        # Execute the actual input truncation, not a test copy of it.
                        exec(compile(cells(name)[0], name, 'exec'), namespace)
                        self.assertEqual(len(namespace['metrics_per_acq']['toy']['LCB']['final test error'][0]), 201)
                    namespace.update(lambdas=[1e-4], dataset_names=['toy'], lcbench_datasets=['toy'])
                    prefix = ('# Cache every' if name == 'NATS_SSS_KnownCost' else
                              '# Pre-compute' if name == 'LCBench_NATS_KnownCost' else
                              '# Save per-combination')
                    source = next(s for s in cells(name) if s.startswith(prefix))
                    exec(compile(source, name, 'exec'), namespace)
                for acquisition, indices in expected.items():
                    acq = 'PBGI' if acquisition.startswith('PBGI(') else acquisition
                    for rule_name, wanted in [('PBGI', indices), ('Immediate', [0, 0, 0]), ('Hindsight', [0, 0, 0])]:
                        if name == 'LCBench_NATS_KnownCost':
                            # Regret equals the iteration, so this also verifies the
                            # real cache's fallback index without adding a test hook.
                            actual = list(namespace['cost_adjusted_regrets']['1e-4']['toy'][acq][rule_name].values())
                        elif name == 'NATS_SSS_KnownCost':
                            key = ('1e-4', 'toy', acq, rule_name, 'estimated cumulative cost')
                            actual = namespace['nats_stopping_results']['stopping_indices'][key]
                        else:
                            cost_key = 'cumulative cost' if 'Unknown' in name else 'estimated cumulative cost'
                            actual = namespace['summary']['results']['1e-4', 'toy', acq, rule_name, cost_key]['stop_indices']
                        self.assertEqual(actual, wanted, (acq, rule_name))


if __name__ == '__main__':
    unittest.main()
