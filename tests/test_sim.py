import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from goodput import analyze, scenario, sim
from goodput.checkpoint import daly

ROOT = Path(__file__).parent.parent
BASE = ROOT / "scenarios" / "01-baseline.json"


class Scenarios(unittest.TestCase):
    def test_all_scenario_files_load(self):
        names = [s.name for s in scenario.load_dir(ROOT / "scenarios")]
        self.assertEqual(names[0], "baseline")
        self.assertEqual(len(names), len(set(names)))

    def test_base_inheritance_overrides_only_given_fields(self):
        s = scenario.load(ROOT / "scenarios" / "03-async-checkpoint.json")
        self.assertEqual(s.checkpoint_minutes, 0.5)
        self.assertEqual(s.job_nodes, 128)
        self.assertAlmostEqual(s.interval_hours(), daly(0.5 / 60, s.nominal_mtbi_hours()))

    def test_unknown_fields_and_bad_values_are_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps({"base": str(BASE), "name": "x", "typo": 1}))
            with self.assertRaisesRegex(ValueError, "typo"):
                scenario.load(p)
            p.write_text(json.dumps({"base": str(BASE), "name": "x", "job_nodes": 500}))
            with self.assertRaisesRegex(ValueError, "the job needs 500"):
                scenario.load(p)

    def test_nominal_mtbi(self):
        s = scenario.load(BASE)
        self.assertAlmostEqual(s.nominal_mtbi_hours(), 1 / (128 / 3000 + 1 / 40))


class Simulation(unittest.TestCase):
    def setUp(self):
        self.s = scenario.load(BASE)

    def test_same_seed_same_log(self):
        a = [e.to_json() for e in sim.simulate(self.s, 7)]
        b = [e.to_json() for e in sim.simulate(self.s, 7)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, [e.to_json() for e in sim.simulate(self.s, 8)])

    def test_log_covers_the_whole_horizon(self):
        m = analyze.summarize(analyze.build_jobs(sim.simulate(self.s, 3)))
        self.assertAlmostEqual(m.total_gpu_hours, self.s.days * 24 * 1024, delta=1)
        self.assertGreater(m.interruptions, 20)

    def test_interruption_rate_matches_the_model(self):
        runs = [analyze.summarize(analyze.build_jobs(sim.simulate(self.s, seed))) for seed in range(1, 31)]
        mtbi = sum(r.running_hours for r in runs) / sum(r.interruptions for r in runs)
        self.assertAlmostEqual(mtbi, self.s.nominal_mtbi_hours(), delta=0.1 * self.s.nominal_mtbi_hours())

    def test_without_spares_hardware_failures_wait_for_repair(self):
        s = scenario.load(ROOT / "scenarios" / "05-no-spares.json")
        m = analyze.summarize(analyze.build_jobs(sim.simulate(s, 1)))
        self.assertGreater(m.gpu_hours["wait"], m.gpu_hours["restart"])

    def test_bad_nodes_show_up_as_repeat_offenders(self):
        s = scenario.load(ROOT / "scenarios" / "06-firmware-regression.json")
        bad = {n.name for n in sim.node_inventory(s) if n.label == "fw-batch-b"}
        counts = analyze.summarize(analyze.build_jobs(sim.simulate(s, 2))).nodes
        top = [n for n, _ in counts.most_common(5)]
        self.assertGreaterEqual(sum(n in bad for n in top), 3)

    def test_job_level_only_failures(self):
        s = replace(self.s, node_groups=[scenario.NodeGroup(132, 1e12)], days=7)
        m = analyze.summarize(analyze.build_jobs(sim.simulate(s, 1)))
        self.assertEqual(m.node_interruptions, 0)
        self.assertGreater(m.interruptions, 0)


if __name__ == "__main__":
    unittest.main()
