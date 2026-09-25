import unittest

import torch

from calibration import apply_temperature, build_calibration, fit_temperature


class CalibrationTests(unittest.TestCase):
    def test_temperature_is_positive_and_bounded(self):
        logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0]])
        targets = torch.tensor([0, 1, 0])
        temperature = fit_temperature(logits, targets)
        self.assertGreaterEqual(temperature, 0.25)
        self.assertLessEqual(temperature, 10.0)

    def test_build_and_apply_calibration(self):
        values = {
            "outfit": {
                "logits": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
                "targets": torch.tensor([0, 1]),
            }
        }
        calibration = build_calibration(values)
        calibrated = apply_temperature(
            values["outfit"]["logits"],
            "outfit",
            calibration,
        )
        self.assertEqual(calibrated.shape, values["outfit"]["logits"].shape)
        self.assertIn("suggested_threshold", calibration["tasks"]["outfit"])


if __name__ == "__main__":
    unittest.main()
