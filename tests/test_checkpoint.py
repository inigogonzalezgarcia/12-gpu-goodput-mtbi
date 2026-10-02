import math
import unittest

from goodput.checkpoint import daly, expected_goodput, young


class CheckpointFormulas(unittest.TestCase):
    def test_young(self):
        self.assertAlmostEqual(young(0.1, 20), math.sqrt(4))

    def test_daly_is_close_to_young_when_checkpoints_are_cheap(self):
        c, m = 5 / 60, 14.8
        self.assertAlmostEqual(daly(c, m) / young(c, m), 1, delta=0.05)
        self.assertLess(daly(c, m), young(c, m))

    def test_daly_caps_at_mtbi_for_huge_checkpoints(self):
        self.assertEqual(daly(10, 4), 4)

    def test_expected_goodput_peaks_near_daly(self):
        c, m, o = 5 / 60, 14.8, 0.5
        grid = [i / 60 for i in range(5, 400)]
        best = max(grid, key=lambda t: expected_goodput(m, c, t, o))
        self.assertAlmostEqual(best, daly(c, m), delta=0.1 * daly(c, m))

    def test_rejects_bad_inputs(self):
        for args in ((0, 1), (1, 0), (-1, 5)):
            with self.assertRaises(ValueError):
                young(*args)
        with self.assertRaises(ValueError):
            expected_goodput(10, 0.1, 0, 0)


if __name__ == "__main__":
    unittest.main()
