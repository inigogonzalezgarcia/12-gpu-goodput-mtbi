import unittest
from pathlib import Path

from goodput import experiments, scenario

ROOT = Path(__file__).parent.parent


class Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = scenario.load(ROOT / "scenarios" / "01-baseline.json")
        cls.sweep = experiments.sweep(cls.base, 12, (15, 30, 60, 90, 120, 240))

    def test_daly_interval_is_as_good_as_the_best_sampled_interval(self):
        self.assertGreaterEqual(self.sweep.daly_simulated, self.sweep.best.simulated - 0.005)

    def test_first_order_model_tracks_the_simulation(self):
        for p in self.sweep.points:
            self.assertAlmostEqual(p.model, p.simulated, delta=0.025, msg=p.interval_minutes)

    def test_each_improvement_moves_the_waste_it_targets(self):
        results = {r.scenario.name: r for r in experiments.compare(scenario.load_dir(ROOT / "scenarios"), 8)}
        b = results["baseline"]
        self.assertLess(results["daly-interval"].shares["checkpoint"], b.shares["checkpoint"])
        self.assertLess(results["fast-hang-detection"].shares["detect"], b.shares["detect"])
        self.assertGreater(results["no-spares"].shares["wait"], b.shares["wait"])
        self.assertLess(results["firmware-regression"].mtbi_hours, b.mtbi_hours)
        self.assertEqual(max(results.values(), key=lambda r: r.mean).scenario.name, "combined")

    def test_quantiles(self):
        self.assertEqual(experiments._quantile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(experiments._quantile([0, 10], 0.1), 1)


if __name__ == "__main__":
    unittest.main()
