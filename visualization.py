import os
import logging

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
# Plot Regimes
# =========================================================
def plot_regimes(features):

    # -----------------------------------------------------
    # Reconstruct cumulative SPY curve
    # -----------------------------------------------------
    spy_returns = features["returns_SPY"]

    cumulative_spy = (
        1 + spy_returns
    ).cumprod()

    plt.figure(figsize=(14, 6))

    plt.plot(
        cumulative_spy.index,
        cumulative_spy,
        label="SPY"
    )

    plt.title(
        "SPY Cumulative Performance"
    )

    plt.xlabel("Date")

    plt.ylabel("Cumulative Return")

    plt.legend()

    plt.tight_layout()

    plt.show()

# =========================================================
# Plot Transition Risk
# =========================================================
def plot_transition_probabilities(
    probabilities
):

    plt.figure(figsize=(14, 5))

    plt.plot(
        probabilities.index,
        probabilities[
            "transition_probability"
        ]
    )

    plt.axhline(
        0.30,
        linestyle="--"
    )

    plt.title(
        "Predicted Transition Risk"
    )

    plt.xlabel("Date")

    plt.ylabel("Probability")

    plt.tight_layout()

    plt.show()


# =========================================================
# Main
# =========================================================
def main():

    features, probabilities = load_data()

    plot_regimes(features)

    plot_transition_probabilities(
        probabilities
    )


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()