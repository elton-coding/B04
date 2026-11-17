import unittest

import pandas as pd
import torch

from geometry import corroborate_emissions, nearby_emissions
from predict import batches_with_context


class ArrivalTests(unittest.TestCase):
    def test_reference_checks_frequency_across_entire_time_interval(self):
        dtype = torch.float64
        actual = nearby_emissions(
            torch.tensor([1000, 1050, 1070, 1090], dtype=dtype),
            torch.tensor([900, 900, 900, 900], dtype=dtype),
            torch.tensor([999.2, 999.9, 1000.1, 1051, 1071.01, 1090], dtype=dtype),
            torch.tensor([900, 500, 500, 910, 900, 911], dtype=dtype),
        )
        self.assertEqual(actual.tolist(), [True, True, False, False])

    def test_boundary_peers_can_corroborate_without_duplicating_core_rows(self):
        first = pd.DataFrame({
            "pulse_id": ["left", "same_platform"],
            "platform_id": ["A", "A"], "toa": [255_999.0, 255_999.0],
            "param1": [900.0, 915.0], "delay": [2.0, 2.0],
        })
        second = pd.DataFrame({
            "pulse_id": ["right"], "platform_id": ["B"], "toa": [256_001.0],
            "param1": [900.0], "delay": [4.0],
        })
        result = {}
        for batch, start, stop in batches_with_context(iter([first, second]), 256_000):
            delay = torch.tensor(batch.delay.to_numpy(copy=True), dtype=torch.float64)[:, None]
            eligible = torch.ones_like(delay, dtype=torch.bool)
            evidence = corroborate_emissions(batch, delay, eligible, torch.device("cpu"))
            for pulse, matched in zip(batch.pulse_id.iloc[start:stop], evidence[start:stop, 0].tolist()):
                self.assertNotIn(pulse, result)
                result[pulse] = matched
        self.assertEqual(result, {"left": True, "same_platform": False, "right": True})


if __name__ == "__main__":
    unittest.main()
