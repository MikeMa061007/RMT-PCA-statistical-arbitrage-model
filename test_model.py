import unittest

import numpy as np
import pandas as pd

import rmt_model
from backtest import calculate_metrics, run_backtest


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
        z_score = np.array([2.0, -1.5, 0.5, 3.0])
        half_life = np.array([20.0, 30.0, 10.0, 200.0])
        signal = rmt_model.build_statistical_arbitrage_signal(z_score, half_life)
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


class BacktestTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        returns = rng.normal(0.0002, 0.01, size=(170, 8))
        prices = 100 * np.cumprod(1 + returns, axis=0)
        self.prices = pd.DataFrame(
            prices,
            index=pd.date_range("2020-01-01", periods=170, freq="B"),
            columns=[f"Asset_{i}" for i in range(8)],
        )

    def test_metric_values_are_finite(self):
        metrics = calculate_metrics([0.01, -0.005, 0.002], [0.2, 0.1, 0.0])
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_walk_forward_output(self):
        daily, weights, metrics = run_backtest(
            self.prices,
            train_window=80,
            rebalance_frequency=10,
            entry_threshold=0.0,
        )
        self.assertEqual(len(daily), 89)
        self.assertEqual(weights.shape, (89, 8))
        self.assertLess(np.abs(weights.sum(axis=1)).max(), 1e-10)
        self.assertIn("maximum_drawdown", metrics)

    def test_last_return_does_not_change_last_position(self):
        first = run_backtest(self.prices, train_window=80, entry_threshold=0.0)[1]
        changed = self.prices.copy()
        changed.iloc[-1, 0] *= 2
        second = run_backtest(changed, train_window=80, entry_threshold=0.0)[1]
        np.testing.assert_allclose(first.iloc[-1], second.iloc[-1])


if __name__ == "__main__":
    unittest.main()
