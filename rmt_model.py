from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf


def generate_data(
    n_samples=400,
    n_variables=30,
    n_factors=3,
    distribution="student_t",
    seed=42,
):
    rng = np.random.default_rng(seed)

    if distribution == "gaussian":
        factor_noise = rng.normal(size=(n_samples, n_factors))
        residual_noise = rng.normal(size=(n_samples, n_variables))
    elif distribution == "student_t":
        factor_noise = rng.standard_t(5, size=(n_samples, n_factors)) / np.sqrt(5 / 3)
        residual_noise = rng.standard_t(5, size=(n_samples, n_variables)) / np.sqrt(5 / 3)
    else:
        raise ValueError("distribution must be gaussian or student_t")

    loadings = rng.normal(size=(n_variables, n_factors))
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)
    common_part = 0.5 * factor_noise @ loadings.T
    residual_part = np.zeros((n_samples, n_variables))

    for t in range(1, n_samples):
        residual_part[t] = 0.8 * residual_part[t - 1] + 0.25 * residual_noise[t]

    return common_part + residual_part


def standardize_data(data, robust=True, winsor_ratio=0.01):
    x = np.asarray(data, dtype=float).copy()

    if winsor_ratio > 0:
        lower = np.percentile(x, 100 * winsor_ratio, axis=0)
        upper = np.percentile(x, 100 * (1 - winsor_ratio), axis=0)
        x = np.clip(x, lower, upper)

    if robust:
        center = np.median(x, axis=0)
        scale = 1.4826 * np.median(np.abs(x - center), axis=0)
    else:
        center = np.mean(x, axis=0)
        scale = np.std(x, axis=0, ddof=1)

    scale[scale < 1e-10] = 1.0
    return (x - center) / scale


def mp_bounds(n_samples, n_variables):
    if n_samples <= n_variables:
        raise ValueError("n_samples must be greater than n_variables")

    alpha = n_samples / n_variables
    lower = (1 - 1 / np.sqrt(alpha)) ** 2
    upper = (1 + 1 / np.sqrt(alpha)) ** 2
    return lower, upper


def bootstrap_upper_edge(data, repeats=30, block_size=10, seed=42):
    x = np.asarray(data, dtype=float)
    n_samples, n_variables = x.shape
    rng = np.random.default_rng(seed)
    maximum_eigenvalues = []
    blocks_needed = int(np.ceil(n_samples / block_size))

    for _ in range(repeats):
        shuffled = np.empty_like(x)

        for column in range(n_variables):
            starts = rng.integers(0, n_samples, size=blocks_needed)
            indices = []

            for start in starts:
                indices.extend((start + np.arange(block_size)) % n_samples)

            shuffled[:, column] = x[np.asarray(indices[:n_samples]), column]

        correlation = np.corrcoef(shuffled, rowvar=False)
        maximum_eigenvalues.append(np.linalg.eigvalsh(correlation)[-1])

    return np.percentile(maximum_eigenvalues, 95)


def pca_residuals(data, upper_edge):
    x = np.asarray(data, dtype=float)
    correlation = x.T @ x / x.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    n_factors = min(int(np.sum(eigenvalues > upper_edge)), x.shape[1] - 1)
    factor_vectors = eigenvectors[:, :n_factors]

    if n_factors == 0:
        residuals = x.copy()
    else:
        residuals = x - x @ factor_vectors @ factor_vectors.T

    return residuals, eigenvalues, factor_vectors, n_factors


def estimate_ar1(cumulative_residuals):
    states = np.asarray(cumulative_residuals, dtype=float)
    x = states[:-1]
    y = states[1:]
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    denominator = np.sum(x_centered**2, axis=0)
    coefficient = np.divide(
        np.sum(x_centered * y_centered, axis=0),
        denominator,
        out=np.zeros(states.shape[1]),
        where=denominator > 1e-12,
    )
    intercept = y.mean(axis=0) - coefficient * x.mean(axis=0)
    valid = (coefficient > 0) & (coefficient < 1)
    long_mean = np.full(states.shape[1], np.nan)
    half_life = np.full(states.shape[1], np.inf)
    long_mean[valid] = intercept[valid] / (1 - coefficient[valid])
    half_life[valid] = np.log(2) / (-np.log(coefficient[valid]))
    fitted = intercept + coefficient * x
    error_std = np.std(y - fitted, axis=0, ddof=2)
    stationary_std = np.full(states.shape[1], np.nan)
    stationary_std[valid] = error_std[valid] / np.sqrt(1 - coefficient[valid] ** 2)
    z_score = (states[-1] - long_mean) / stationary_std
    z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)

    return coefficient, long_mean, half_life, z_score


def clean_covariance(data, upper_edge, method="clip"):
    x = np.asarray(data, dtype=float)

    if method == "shrinkage":
        return LedoitWolf().fit(x).covariance_

    scale = np.std(x, axis=0, ddof=1)
    scale[scale < 1e-10] = 1.0
    correlation = np.corrcoef(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    noise = eigenvalues <= upper_edge

    if np.any(noise):
        eigenvalues[noise] = np.mean(eigenvalues[noise])

    cleaned_correlation = (eigenvectors * eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(cleaned_correlation))
    cleaned_correlation /= np.outer(diagonal, diagonal)
    return cleaned_correlation * np.outer(scale, scale)



def build_statistical_arbitrage_signal(
    z_score,
    half_life,
    entry_threshold=1.0,
    min_half_life=2.0,
    max_half_life=120.0,
):
    z_score = np.asarray(z_score, dtype=float)
    half_life = np.asarray(half_life, dtype=float)
    eligible = (
        np.isfinite(z_score)
        & np.isfinite(half_life)
        & (np.abs(z_score) >= entry_threshold)
        & (half_life >= min_half_life)
        & (half_life <= max_half_life)
    )
    signal = np.zeros_like(z_score)
    signal[eligible] = -z_score[eligible]
    if np.count_nonzero(eligible) > 1:
        signal[eligible] -= signal[eligible].mean()
    gross_exposure = np.abs(signal).sum()
    if gross_exposure > 0:
        signal /= gross_exposure
    return signal


def run_rmt_model(data, robust=True, use_bootstrap=True, covariance_method="clip", seed=42):
    standardized = standardize_data(data, robust=robust)
    mp_lower, mp_upper = mp_bounds(*standardized.shape)
    selected_edge = (
        bootstrap_upper_edge(standardized, seed=seed)
        if use_bootstrap
        else mp_upper
    )
    residuals, eigenvalues, factor_vectors, n_factors = pca_residuals(
        standardized,
        selected_edge,
    )
    cumulative_residuals = np.cumsum(residuals, axis=0)
    coefficient, long_mean, half_life, z_score = estimate_ar1(cumulative_residuals)
    cleaned_covariance = clean_covariance(data, selected_edge, covariance_method)
    statistical_arbitrage_signal = build_statistical_arbitrage_signal(z_score, half_life)
    covariance_condition = np.linalg.cond(cleaned_covariance)

    return {
        "mp_lower": mp_lower,
        "mp_upper": mp_upper,
        "selected_edge": selected_edge,
        "eigenvalues": eigenvalues,
        "factor_vectors": factor_vectors,
        "n_factors": n_factors,
        "residuals": residuals,
        "cumulative_residuals": cumulative_residuals,
        "ar_coefficient": coefficient,
        "long_mean": long_mean,
        "half_life": half_life,
        "statistical_arbitrage_signal": statistical_arbitrage_signal,
        "z_score": z_score,
        "cleaned_covariance": cleaned_covariance,
        "covariance_condition": covariance_condition,
    }


def save_result(result, output="results/rmt_model_result.png"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    eigenvalues = result["eigenvalues"]
    z_score = result["z_score"]
    covariance = result["cleaned_covariance"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, "o-", markersize=4)
    axes[0].axhline(result["mp_upper"], color="tab:orange", linestyle="--", label="MP upper")
    axes[0].axhline(result["selected_edge"], color="tab:red", linestyle=":", label="Bootstrap edge")
    axes[0].set_title("Correlation eigenvalues")
    axes[0].set_xlabel("Index")
    axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(len(z_score)), z_score)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residual z-scores")
    axes[1].set_xlabel("Variable")
    image = axes[2].imshow(covariance, cmap="coolwarm", aspect="auto")
    axes[2].set_title("Cleaned covariance")
    axes[2].set_xlabel("Variable")
    axes[2].set_ylabel("Variable")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    data = generate_data()
    result = run_rmt_model(data)
    save_result(result)
    finite_half_life = result["half_life"][np.isfinite(result["half_life"])]
    print("data shape:", data.shape)
    print("MP upper edge:", round(result["mp_upper"], 4))
    print("bootstrap edge:", round(result["selected_edge"], 4))
    print("selected factors:", result["n_factors"])
    print("active statistical arbitrage signals:", np.count_nonzero(result["statistical_arbitrage_signal"]))
    print("median residual half-life:", round(np.median(finite_half_life), 2))
    print("covariance condition number:", round(result["covariance_condition"], 2))
