import unittest

import numpy as np

from utils import find_matching_frames


class FindMatchingFramesTests(unittest.TestCase):
    def test_matches_only_iris_frames_inside_requested_window(self):
        iris_times = np.array([90.0, 100.0, 110.0, 130.0])
        aia_times = np.array([99.0, 112.0])

        matches = find_matching_frames(
            iris_times, aia_times, start_s=100.0, end_s=120.0, delta_t=3.0
        )

        self.assertEqual(matches, [(0, 1), (1, 2)])

    def test_rejects_pair_outside_time_tolerance(self):
        matches = find_matching_frames(
            np.array([100.0]),
            np.array([104.0]),
            start_s=100.0,
            end_s=100.0,
            delta_t=3.0,
        )

        self.assertEqual(matches, [])

    def test_includes_pair_on_time_tolerance_boundary(self):
        matches = find_matching_frames(
            np.array([100.0]),
            np.array([103.0]),
            start_s=100.0,
            end_s=100.0,
            delta_t=3.0,
        )

        self.assertEqual(matches, [(0, 0)])

    def test_returns_no_matches_when_no_aia_frames_are_available(self):
        matches = find_matching_frames(
            np.array([100.0]),
            np.array([]),
            start_s=100.0,
            end_s=100.0,
        )

        self.assertEqual(matches, [])

    def test_rejects_negative_time_tolerance(self):
        with self.assertRaisesRegex(ValueError, "delta_t must be non-negative"):
            find_matching_frames(
                np.array([100.0]),
                np.array([100.0]),
                start_s=100.0,
                end_s=100.0,
                delta_t=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
