import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# Load Data
# =========================================================
def load_data():

    BASE_DIR = os.path.dirname(
        os.path.dirname(__file__)
    )

    features_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "features.csv"
    )

    probabilities_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "transition_probabilities.csv"
    )

    features = pd.read_csv(
        features_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    probabilities = pd.read_csv(
        probabilities_path,
        index_col=0,
        parse_dates=True
    )

    # -----------------------------------------------------
    # Flatten columns
    # -----------------------------------------------------
    features.columns = [
        f"{col[0]}_{col[1]}"
        for col in features.columns
    ]

    return features, probabilities


# =========================================================
# Create Strategy Returns
# =========================================================
def create_strategy_returns(
    features,
    probabilities,
    threshold=0.30
):

    logging.info(
        "Creating adaptive allocation strategy"
    )

    # -----------------------------------------------------
    # Market returns
    # -----------------------------------------------------
    spy_returns = features["returns_SPY"]

    # -----------------------------------------------------
    # Defensive asset
    # -----------------------------------------------------
    gld_returns = features["returns_GLD"]

    # -----------------------------------------------------
    # Transition probabilities
    # -----------------------------------------------------
    probabilities = probabilities.squeeze()

    # -----------------------------------------------------
    # Align dates
    # -----------------------------------------------------
    spy_returns = spy_returns.loc[
        probabilities.index
    ]

    gld_returns = gld_returns.loc[
        probabilities.index
    ]

    # -----------------------------------------------------
    # Risk signal
    # -----------------------------------------------------
    risk_off = (
        probabilities > threshold
    ).astype(int)

    # -----------------------------------------------------
    # Dynamic allocations
    # -----------------------------------------------------
    spy_weight = np.where(
        risk_off == 1,
        0.6,
        1.0
    )

    gld_weight = np.where(
        risk_off == 1,
        0.4,
        0.0
    )

    # -----------------------------------------------------
    # Portfolio returns
    # -----------------------------------------------------
    strategy_returns = (
        spy_weight * spy_returns
        +
        gld_weight * gld_returns
    )

    # -----------------------------------------------------
    # Benchmark
    # -----------------------------------------------------
    benchmark_returns = spy_returns

    return (
        strategy_returns,
        benchmark_returns
    )
# =========================================================
# Plot Performance
# =========================================================
def plot_performance(
    strategy_returns,
    benchmark_returns
):

    strategy_curve = (
        1 + strategy_returns
    ).cumprod()

    benchmark_curve = (
        1 + benchmark_returns
    ).cumprod()

    plt.figure(figsize=(12, 6))

    plt.plot(
        strategy_curve,
        label="Adaptive Strategy"
    )

    plt.plot(
        benchmark_curve,
        label="Buy and Hold"
    )

    plt.title(
        "Strategy vs Benchmark"
    )

    plt.xlabel("Time")

    plt.ylabel("Portfolio Value")

    plt.legend()

    plt.tight_layout()

    plt.show()


# =========================================================
# Performance Metrics
# =========================================================
def performance_metrics(
    strategy_returns,
    benchmark_returns
):

    strategy_sharpe = (
        strategy_returns.mean()
        /
        strategy_returns.std()
    ) * np.sqrt(252)

    benchmark_sharpe = (
        benchmark_returns.mean()
        /
        benchmark_returns.std()
    ) * np.sqrt(252)

    logging.info(
        f"Strategy Sharpe Ratio: "
        f"{strategy_sharpe:.2f}"
    )

    logging.info(
        f"Benchmark Sharpe Ratio: "
        f"{benchmark_sharpe:.2f}"
    )


# =========================================================
# Main
# =========================================================
def main():

    features, probabilities = load_data()

    (
        strategy_returns,
        benchmark_returns
    ) = create_strategy_returns(
        features,
        probabilities
    )

    performance_metrics(
        strategy_returns,
        benchmark_returns
    )

    plot_performance(
        strategy_returns,
        benchmark_returns
    )


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()