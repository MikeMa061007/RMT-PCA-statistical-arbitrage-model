# RMT-PCA Statistical Arbitrage Model

This project implements a compact statistical arbitrage model for a high-dimensional return matrix. Random Matrix Theory separates noisy correlation eigenvalues from common market structure, PCA removes the common factors, and an AR(1) model measures mean reversion in the remaining residual components. Residual deviations are converted into relative long and short signals, while a cleaned covariance matrix supports financial risk analysis.

## Model structure

Let the return matrix be

```text
X ∈ R^(m×n)
```

where `m` is the number of observations and `n` is the number of financial variables. The synthetic example combines a
small number of common factors with autocorrelated residual components. Student-t innovations are used by default
to represent heavier tails than Gaussian data.

### 1. Robust standardization

Each column is standardized before the correlation matrix is estimated. The standard version uses the sample mean
and standard deviation. The robust version uses the median and median absolute deviation:

```text
scale = 1.4826 × median(|X - median(X)|)
```

Winsorization limits the influence of a small number of extreme observations.

### 2. Random Matrix Theory boundary

For standardized data, the sample correlation matrix is

```text
C = X'X / m
```

When `α = m/n > 1`, the Marchenko-Pastur noise interval is

```text
λ- = (1 - 1/√α)^2
λ+ = (1 + 1/√α)^2
```

The model also estimates an empirical upper edge with a circular block bootstrap. Each variable is resampled
independently in consecutive time blocks. This retains short-term dependence within a variable while removing the
original cross-sectional alignment. Eigenvalues above the selected upper edge are treated as common statistical
factors.

### 3. PCA residual model

Let `V` contain the selected eigenvectors. The common component and residual matrix are

```text
X_common = XVV'
E = X - XVV'
```

The residual matrix is orthogonal to the selected PCA factor space. This separates broad common movement from the
variable-specific component used in the next stage.

### 4. AR(1) mean-reversion measurement

The residuals are accumulated through time to form `S`. Each column is fitted with

```text
S(t+1) = a + bS(t) + error(t+1)
```

For `0 < b < 1`, the fitted process has the following long-run mean and half-life:

```text
long_mean = a / (1-b)
half_life = log(2) / -log(b)
```

The residual z-score measures the current state relative to its fitted long-run level and stationary variation.

### 5. Statistical arbitrage signal

The residual z-score measures how far each variable is from its estimated equilibrium. A variable is used when its absolute z-score is at least 1 and its estimated half-life is between 2 and 120 periods.

The raw signal is the opposite of the residual z-score:

```text
signal = -residual z-score
```

The active signals are centered and normalized so that positive and negative positions represent relative long and short directions. This is a simple model signal, not a complete trading or profit evaluation system.

### 6. Covariance cleaning

Two covariance estimators are available:

- MP eigenvalue clipping replaces eigenvalues inside the noise region with their average.
- Ledoit-Wolf shrinkage moves the sample covariance toward a better-conditioned target.

The condition number of the cleaned covariance matrix is reported as a numerical stability diagnostic.

## Complete process

```text
Return matrix
→ robust standardization
→ correlation eigendecomposition
→ MP or bootstrap spectral boundary
→ PCA common factors
→ PCA residuals
→ AR(1) parameters and residual z-scores
→ statistical arbitrage signals
→ cleaned covariance and condition number
```

## Run the model

```bash
python rmt_model.py
```

The program prints the spectral boundaries, selected factor count, active statistical arbitrage signal count, residual half-life and covariance condition number. It also saves one figure containing the eigenvalue spectrum, residual z-scores and cleaned covariance matrix.

## Run the tests

```bash
python -m unittest -v
```

The tests check the MP formula, PCA residual orthogonality, statistical arbitrage signal normalization, covariance positive definiteness and the complete model output shapes.

