from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmt_model import build_statistical_arbitrage_signal, run_rmt_model


def load_prices(path="data/historical_prices.csv"):
    prices = pd.read_csv(path, index_col="date", parse_dates=True).sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna()
    if prices.shape[1] < 3 or prices.shape[0] < 100:
        raise ValueError("price data must contain at least 100 rows and 3 assets")
    return prices


def calculate_metrics(net_returns, turnover):
    values = np.asarray(net_returns, dtype=float)
    equity = np.cumprod(1 + values)
    years = len(values) / 252
    annual_return = equity[-1] ** (1 / years) - 1 if years > 0 else 0.0
    annual_volatility = np.std(values, ddof=1) * np.sqrt(252)
    sharpe_ratio = (
        np.mean(values) / np.std(values, ddof=1) * np.sqrt(252)
        if np.std(values, ddof=1) > 0
        else 0.0
    )
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1
    return {
        "total_return": equity[-1] - 1,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": drawdown.min(),
        "average_daily_turnover": np.mean(turnover),
        "active_days": int(np.count_nonzero(values)),
    }


def run_backtest(
    prices,
    train_window=120,
    rebalance_frequency=20,
    transaction_cost_bps=5.0,
    entry_threshold=0.75,
    volatility_scaling=True,
):
    returns = prices.copy().sort_index().pct_change(fill_method=None).dropna()
    if train_window <= returns.shape[1] or train_window >= len(returns):
        raise ValueError("train_window must exceed the asset count and be shorter than the data")

    weights = np.zeros((len(returns), returns.shape[1]))
    gross_returns = np.zeros(len(returns))
    net_returns = np.zeros(len(returns))
    turnover = np.zeros(len(returns))
    costs = np.zeros(len(returns))
    factor_count = np.zeros(len(returns), dtype=int)
    previous = np.zeros(returns.shape[1])
    cost_rate = transaction_cost_bps / 10000

    for t in range(train_window, len(returns)):
        if (t - train_window) % rebalance_frequency == 0:
            training = returns.iloc[t - train_window : t].to_numpy()
            model = run_rmt_model(training, use_bootstrap=False, seed=t)
            target = build_statistical_arbitrage_signal(
                model["z_score"],
                model["half_life"],
                entry_threshold=entry_threshold,
                min_half_life=2.0,
                max_half_life=60.0,
            )
            if volatility_scaling:
                volatility = np.std(training, axis=0, ddof=1)
                target = np.divide(target, volatility, out=np.zeros_like(target), where=volatility > 0)
            vectors = model["factor_vectors"]
            if vectors.size:
                target = target - vectors @ (vectors.T @ target)
            target = target - target.mean()
            gross = np.abs(target).sum()
            if gross > 0:
                target = target / gross
            previous = target
            factor_count[t] = model["n_factors"]
        else:
            factor_count[t] = factor_count[t - 1]

        weights[t] = previous
        turnover[t] = np.abs(weights[t] - weights[t - 1]).sum()
        costs[t] = turnover[t] * cost_rate
        gross_returns[t] = weights[t] @ returns.iloc[t].to_numpy()
        net_returns[t] = gross_returns[t] - costs[t]

    index = returns.index[train_window:]
    daily = pd.DataFrame(
        {
            "gross_return": gross_returns[train_window:],
            "cost": costs[train_window:],
            "net_return": net_returns[train_window:],
            "turnover": turnover[train_window:],
            "gross_exposure": np.abs(weights[train_window:]).sum(axis=1),
            "net_exposure": weights[train_window:].sum(axis=1),
            "factor_count": factor_count[train_window:],
        },
        index=index,
    )
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    weight_frame = pd.DataFrame(weights[train_window:], index=index, columns=returns.columns)
    metrics = calculate_metrics(daily["net_return"], daily["turnover"])
    return daily, weight_frame, metrics


def save_backtest(daily, weights, metrics, output_directory="results"):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "backtest_daily.csv", index_label="date")
    weights.to_csv(output / "backtest_weights.csv", index_label="date")
    pd.Series(metrics, name="value").to_csv(output / "backtest_metrics.csv", header=True)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(daily.index, daily["equity"], color="tab:blue")
    axes[0].set_title("2019 walk-forward backtest")
    axes[0].set_ylabel("Growth of $1")
    axes[1].fill_between(daily.index, daily["drawdown"], 0, color="tab:red", alpha=0.4)
    axes[1].set_title("Strategy drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    figure.tight_layout()
    figure.savefig(output / "historical_backtest.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    prices = load_prices()
    daily, weights, metrics = run_backtest(prices)
    save_backtest(daily, weights, metrics)
    print("backtest period:", daily.index.min().date(), "to", daily.index.max().date())
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
