import unittest

import numpy as np

import rmt_model


class RMTModelTest(unittest.TestCase):
    def test_mp_bounds(self):
        lower, upper = rmt_model.mp_bounds(400, 200)
        self.assertAlmostEqual(lower, (1 - 1 / np.sqrt(2)) ** 2)
        self.assertAlmostEqual(upper, (1 + 1 / np.sqrt(2)) ** 2)

    def test_pca_residuals(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(120, 10))
        residuals, _, vectors, _ = rmt_model.pca_residuals(data, 1.2)
        np.testing.assert_allclose(residuals @ vectors, 0.0, atol=1e-10)

    def test_covariance_cleaning(self):
        data = rmt_model.generate_data(n_samples=160, n_variables=10, seed=3)
        covariance = rmt_model.clean_covariance(data, upper_edge=1.5)
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)

    def test_statistical_arbitrage_signal(self):
        from rmt_model import build_statistical_arbitrage_signal

        z_score = np.array([2.0, -1.5, 0.5, 3.0])
        half_life = np.array([20.0, 30.0, 10.0, 200.0])
        signal = build_statistical_arbitrage_signal(z_score, half_life)
        self.assertAlmostEqual(float(signal.sum()), 0.0)
        self.assertAlmostEqual(float(np.abs(signal).sum()), 1.0)
        self.assertEqual(float(signal[2]), 0.0)
        self.assertEqual(float(signal[3]), 0.0)

    def test_complete_model(self):
        data = rmt_model.generate_data(n_samples=180, n_variables=12, seed=4)
        result = rmt_model.run_rmt_model(data, use_bootstrap=False, seed=4)
        self.assertEqual(result["residuals"].shape, data.shape)
        self.assertEqual(result["cleaned_covariance"].shape, (12, 12))
        self.assertEqual(result["statistical_arbitrage_signal"].shape, (12,))
        self.assertTrue(np.isfinite(result["covariance_condition"]))


if __name__ == "__main__":
    unittest.main()
