import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from goodput.cli import main

ROOT = Path(__file__).parent.parent


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


class CLI(unittest.TestCase):
    def test_simulate_then_analyze(self):
        with tempfile.TemporaryDirectory() as d:
            log = str(Path(d) / "e.jsonl")
            self.assertEqual(run("simulate", str(ROOT / "scenarios/01-baseline.json"), "--seed", "4", "-o", log)[0], 0)
            code, out, _ = run("analyze", log, "--slo", "0.85", "--format", "json")
            self.assertEqual(code, 0)
            d = json.loads(out)
            self.assertGreater(d["interruptions"], 0)
            self.assertEqual(len(d["slo"]["windows"]), 4)

    def test_analyze_sample(self):
        code, out, _ = run("analyze", str(ROOT / "examples/sample-events.jsonl"))
        self.assertEqual(code, 0)
        self.assertIn("Goodput: 53.3%", out)

    def test_checkpoint(self):
        code, out, _ = run("checkpoint", "--mtbi-hours", "14.8", "--checkpoint-minutes", "5")
        self.assertEqual(code, 0)
        self.assertIn("Daly: 90.9 min", out)

    def test_errors_exit_2(self):
        code, _, err = run("analyze", "/nonexistent.jsonl")
        self.assertEqual(code, 2)
        self.assertIn("goodput:", err)

    def test_committed_results_are_reproducible(self):
        code, out, err = run("report", "--scenarios", str(ROOT / "scenarios"), "--check", str(ROOT / "docs/results.md"))
        self.assertEqual(code, 0, err)


if __name__ == "__main__":
    unittest.main()
