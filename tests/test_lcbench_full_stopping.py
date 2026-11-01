"""Regression checks for combinations in the full LCBench analysis notebook.

Only the rule-definition cell runs: no experiment files are loaded or changed.
Run with ``python3 -m unittest discover -s tests -p 'test_lcbench_full_stopping.py'``.
"""

import json
from pathlib import Path
import unittest

import numpy as np


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "LCBench_KnownCost_full.ipynb"
)
RULE_NAMES = (
    "PBGI", "LogEIPC-med", "SRGap-med", "UCB-LCB", "PRB", "GSS", "Convergence"
)


def trajectory(stop):
    """A run with improvement until a plateau, and a crossing at ``stop``."""
    rounds = np.arange(40)
    best = 100.0 - np.minimum(rounds, stop - 5)
    return {
        "current best observed": best[None, :],
        "PBGI(1e-4) acq": np.where(rounds < stop, best - 1, best + 1)[None, :],
        "LogEIPC acq": np.where(rounds < stop, 0.0, -10.0)[None, :],
        "exp min regret gap": np.where(rounds < stop, 10.0, 0.1)[None, :],
        "regret upper bound": np.where(rounds < stop, 1.0, 0.005)[None, :],
        "PRB": np.where(rounds < stop, 0.0, 0.99)[None, :],
    }


class FullNotebookStoppingTests(unittest.TestCase):
    def setUp(self):
        self.expected = {"PBGI(1e-4)": 22, "LogEIPC": 30, "LCB": 26, "TS": 34}
        self.metrics = {"toy": {a: trajectory(t) for a, t in self.expected.items()}}
        notebook = json.loads(NOTEBOOK.read_text())
        definition_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "templates = [" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(definition_cells), 1)
        self.namespace = {"metrics_per_acq": self.metrics}
        exec(compile(definition_cells[0], str(NOTEBOOK), "exec"), self.namespace)
        self.rules = {
            rule["stp_key"]: rule["condition_fn"]
            for rule in self.namespace["stopping_rules"]
            if rule["acq_key"] == "PBGI(1e-4)" and rule["stp_key"] in RULE_NAMES
        }

    def test_every_rule_stops_on_its_selected_trajectory(self):
        self.assertEqual(self.namespace["init"], 20)
        for name, condition in self.rules.items():
            for acquisition, expected in self.expected.items():
                with self.subTest(rule=name, acquisition=acquisition):
                    actual = next(
                        (i for i in range(20, 40)
                         if condition(i, 0, "toy", fa=acquisition)),
                        None,
                    )
                    self.assertEqual(actual, expected)

    def test_trajectory_must_be_explicit(self):
        for name, condition in self.rules.items():
            with self.subTest(rule=name):
                with self.assertRaises(TypeError):
                    condition(22, 0, "toy")

    def test_pbgi_uses_its_metric_and_same_row_incumbent(self):
        run = self.metrics["toy"]["LogEIPC"]
        run["current best observed"][0, 20:22] = [10.0, 9.0]
        run["PBGI(1e-4) acq"][0, 21] = 9.5
        # 9.5 crosses the current incumbent 9, but not the previous incumbent 10.
        # The LogEIPC metric is still 0, so using it as a Gittins index also fails.
        self.assertTrue(self.rules["PBGI"](21, 0, "toy", fa="LogEIPC"))

    def test_median_threshold_uses_the_selected_trajectory(self):
        run = self.metrics["toy"]["LogEIPC"]
        run["LogEIPC acq"][0, 1:21] = 10.0
        run["LogEIPC acq"][0, 21] = 4.0
        run["exp min regret gap"][0, 1:21] = 100.0
        run["exp min regret gap"][0, 21] = 5.0
        for name in ("LogEIPC-med", "SRGap-med"):
            with self.subTest(rule=name):
                self.assertTrue(self.rules[name](21, 0, "toy", fa="LogEIPC"))


if __name__ == "__main__":
    unittest.main()
