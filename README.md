# Regime-Aware Market Transition Forecasting & Adaptive Allocation System

## Overview

This project builds a quantitative machine learning system for:

- detecting hidden market regimes
- forecasting short-horizon regime transition risk
- dynamically adjusting portfolio exposure during elevated instability periods

The system combines:

- Hidden Markov Models (HMM)
- Monte Carlo simulation
- XGBoost
- cross-asset feature engineering
- probabilistic risk forecasting
- adaptive portfolio allocation

The goal is to create an institutional-style framework for market instability detection and risk-aware portfolio management.

---

# Motivation

Financial markets behave differently during:

- calm periods
- high-volatility environments
- stress/crisis regimes

Traditional forecasting approaches often fail to account for these hidden market states.

This project models markets as a regime-switching system and uses machine learning to estimate the probability of near-term instability transitions.

---

# System Architecture

## 1. Market Data Collection

Assets include:

- SPY
- QQQ
- GLD
- BTC-USD
- AAPL
- MSFT
- GOOG
- TCS.NS
- HDFCBANK.NS

Data is downloaded using:

- yfinance

---

## 2. Feature Engineering

Engineered features include:

### Return Features
- daily returns
- log returns

### Momentum Features
- rolling momentum signals

### Volatility Features
- rolling volatility
- volatility clustering indicators

### Mean-Reversion Features
- rolling z-scores

### Cross-Asset Signals
- crypto volatility
- equity momentum
- defensive asset movement

---

# Regime Detection

Hidden Markov Models (Gaussian HMMs) are used to identify latent market regimes.

The model learns hidden states directly from:
- returns
- volatility dynamics

without manually labeling regimes.

Detected regimes represent:
- calm markets
- elevated volatility environments
- stress periods

---

# Monte Carlo Simulation

Monte Carlo simulation is used to model future uncertainty under different market regimes.

The framework simulates:
- future return paths
- volatility scenarios
- regime-sensitive uncertainty

This provides probabilistic risk estimation instead of single-point forecasts.

---

# Machine Learning Forecasting

## Objective

Predict whether a regime transition is likely within the next 5 trading days.

This formulation produced stronger predictive structure than exact next-day transition prediction.

---

## Model

The forecasting model uses:

- XGBoost Classifier

Reasons for selection:
- strong performance on tabular financial data
- robustness to nonlinear relationships
- effective handling of weak noisy signals

---

# Evaluation Methodology

The project uses:

- walk-forward validation
- probabilistic forecasting
- rare-event evaluation metrics

Metrics include:

- ROC-AUC
- PR-AUC
- Sharpe Ratio

Accuracy alone was avoided due to severe class imbalance.

---

# Explainability

SHAP analysis is used to interpret model behavior and identify the most influential transition-risk drivers.

Important drivers included:
- BTC volatility
- cross-asset momentum
- defensive asset behavior

---

# Adaptive Allocation Strategy

The predicted transition probabilities are used as a portfolio risk signal.

## Allocation Logic

### Low Transition Risk
- 100% SPY allocation

### High Transition Risk
- 60% SPY
- 40% GLD

This creates a simple regime-aware defensive allocation overlay.

---

# Results

## Forecasting Performance

- ROC-AUC ≈ 0.68
- PR-AUC ≈ 0.27

The results suggest that:
- exact daily transitions are highly noisy
- short-horizon instability forecasting contains meaningful predictive structure

---

## Portfolio Performance

### Adaptive Strategy Sharpe Ratio
- 1.75

### Buy-and-Hold Benchmark Sharpe Ratio
- 1.28

The adaptive allocation framework improved risk-adjusted performance during elevated instability periods.

---

# Key Insights

- market transitions are difficult to predict at exact daily resolution
- short-horizon instability forecasting is more learnable
- crypto volatility acted as an important stress indicator
- probabilistic forecasting is more useful than binary prediction
- portfolio construction matters as much as prediction quality

---

# Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- hmmlearn
- xgboost
- shap
- yfinance

---

# Future Improvements

Potential future extensions include:

- volatility forecasting
- correlation regime modeling
- macroeconomic feature integration
- transaction cost modeling
- portfolio optimization
- online learning systems

---

# Disclaimer

This project is for educational and research purposes only and does not constitute investment advice.