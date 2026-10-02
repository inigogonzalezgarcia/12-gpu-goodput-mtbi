import io
import unittest
from datetime import datetime, timezone
from pathlib import Path

from goodput import analyze, events

SAMPLE = Path(__file__).parent.parent / "examples" / "sample-events.jsonl"


def log(lines):
    return analyze.build_jobs(events.read(io.StringIO("\n".join(lines))))


def ev(minute, event, **extra):
    import json
    t = f"2026-09-01T{minute // 60:02d}:{minute % 60:02d}:00Z"
    return json.dumps({"time": t, "job": "j", "event": event, **extra})


class SampleLog(unittest.TestCase):
    def setUp(self):
        self.jobs = analyze.build_jobs(events.read(SAMPLE.read_text().splitlines()))
        self.s = analyze.summarize(self.jobs)

    def test_every_hour_is_accounted_for(self):
        # 3.75 h wall-clock on 8 GPUs
        self.assertAlmostEqual(self.s.total_gpu_hours, 3.75 * 8)
        want = {"productive": 2.0, "checkpoint": 0.25, "lost": 0.5, "detect": 0.25, "wait": 0.25, "restart": 0.5}
        for cat, hours in want.items():
            self.assertAlmostEqual(self.s.gpu_hours[cat], hours * 8, msg=cat)
        self.assertAlmostEqual(self.s.goodput, 2 / 3.75)

    def test_cost_is_charged_to_the_cause(self):
        self.assertEqual(self.s.by_cause["gpu_xid79"]["count"], 1)
        self.assertAlmostEqual(self.s.by_cause["gpu_xid79"]["gpu_hours"], 1.5 * 8)  # lost + detect + wait + restart

    def test_mtbi_and_node_mtbf(self):
        self.assertAlmostEqual(self.s.mtbi_hours, 2.75)  # compute + checkpoint time
        self.assertAlmostEqual(self.s.node_mtbf_hours, 5.5)  # 2 nodes
        self.assertEqual(self.s.nodes["r01-n02"], 1)


class EdgeCases(unittest.TestCase):
    def test_interrupt_during_checkpoint_loses_the_pending_work(self):
        s = analyze.summarize(log([ev(0, "start", nodes=1, gpus_per_node=1), ev(60, "checkpoint_start"),
                                   ev(70, "interrupt", cause="host"), ev(80, "detected"), ev(80, "nodes_ready"),
                                   ev(90, "running"), ev(120, "end")]))
        self.assertAlmostEqual(s.gpu_hours["lost"], 1.0)
        self.assertAlmostEqual(s.gpu_hours["checkpoint"], 10 / 60)
        self.assertAlmostEqual(s.gpu_hours["productive"], 0.5)
        self.assertNotIn("node", s.nodes)

    def test_job_level_interruptions_have_no_node(self):
        s = analyze.summarize(log([ev(0, "start", nodes=4), ev(30, "interrupt", cause="software_crash"),
                                   ev(31, "detected"), ev(31, "nodes_ready"), ev(40, "running"), ev(60, "end")]))
        self.assertEqual(s.interruptions, 1)
        self.assertIsNone(s.node_mtbf_hours)

    def test_rejects_impossible_sequences(self):
        bad = [
            [ev(0, "checkpoint_start")],
            [ev(0, "start"), ev(10, "checkpoint_end"), ev(20, "end")],
            [ev(0, "start"), ev(10, "interrupt", cause="x"), ev(20, "running"), ev(30, "end")],
            [ev(0, "start"), ev(10, "checkpoint_start")],
        ]
        for lines in bad:
            with self.assertRaises(ValueError, msg=lines):
                log(lines)

    def test_rejects_unknown_events_and_naive_times(self):
        with self.assertRaises(ValueError):
            events.read(['{"time": "2026-09-01T00:00:00Z", "job": "j", "event": "reboot"}'])
        with self.assertRaises(ValueError):
            events.read(['{"time": "2026-09-01T00:00:00", "job": "j", "event": "start"}'])


class Windows(unittest.TestCase):
    def test_windows_add_up_to_the_whole(self):
        jobs = analyze.build_jobs(events.read(SAMPLE.read_text().splitlines()))
        total = analyze.summarize(jobs)
        parts = analyze.windows(jobs, 1)
        self.assertEqual(len(parts), 4)
        for cat in analyze.CATEGORIES:
            self.assertAlmostEqual(sum(p[2].gpu_hours[cat] for p in parts), total.gpu_hours[cat], msg=cat)
        self.assertEqual(sum(p[2].interruptions for p in parts), 1)

    def test_slo_budget_burn(self):
        jobs = analyze.build_jobs(events.read(SAMPLE.read_text().splitlines()))
        res = analyze.slo(jobs, 0.5, 24)
        row = res.rows[0]
        self.assertAlmostEqual(row["goodput"], 2 / 3.75)
        self.assertAlmostEqual(row["budget_burn"], (1.75 / 3.75) / 0.5)
        self.assertEqual(res.met, 1)
        self.assertEqual(row["top_waste"], "lost")
        with self.assertRaises(ValueError):
            analyze.slo(jobs, 1.5, 24)


class Timestamps(unittest.TestCase):
    def test_parse_and_format_round_trip(self):
        e = events.Event(datetime(2026, 9, 1, 3, tzinfo=timezone.utc), "j", "end")
        back = events.read([e.to_json()])[0]
        self.assertEqual(back.time, e.time)


if __name__ == "__main__":
    unittest.main()
