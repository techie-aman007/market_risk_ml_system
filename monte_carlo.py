import os
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(features_path: str, regimes_path: str):
    logging.info("Loading data")

    features = pd.read_csv(features_path, header=[0, 1], index_col=0, parse_dates=True)
    regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True)

    regimes.columns = ["regime"]


    
    features.columns = [
    f"{col[0]}_{col[1]}" for col in features.columns
    ]


    df = features.join(regimes, how="inner")

    return df


def estimate_regime_params(df: pd.DataFrame):
    logging.info("Estimating regime parameters")

    # Select log return columns dynamically
    log_return_cols = [col for col in df.columns if col.startswith("log_returns")]

    log_returns = df[log_return_cols].mean(axis=1)

    params = {}

    for regime in sorted(df["regime"].unique()):
        subset = log_returns[df["regime"] == regime]

        mu = subset.mean()
        sigma = subset.std()

        params[regime] = (mu, sigma)

        logging.info(f"Regime {regime}: mu={mu:.5f}, sigma={sigma:.5f}")

    return params


def estimate_transition_matrix(regimes: pd.Series):
    logging.info("Estimating transition matrix")

    states = regimes.values
    n_states = len(np.unique(states))

    matrix = np.zeros((n_states, n_states))

    for i in range(len(states) - 1):
        matrix[states[i], states[i + 1]] += 1

    matrix = matrix / matrix.sum(axis=1, keepdims=True)

    logging.info(f"\nTransition Matrix:\n{matrix}")

    return matrix



def simulate_paths(
    params,
    transition_matrix,
    start_price=100,
    n_steps=252,
    n_simulations=1000
):
    logging.info("Running Monte Carlo simulation")

    n_states = len(params)
    simulations = []

    for sim in range(n_simulations):

        prices = [start_price]

        # random initial state
        state = np.random.choice(range(n_states))

        for t in range(n_steps):

            mu, sigma = params[state]

            # simulate return
            r = np.random.normal(mu, sigma)

            new_price = prices[-1] * np.exp(r)
            prices.append(new_price)

            # transition to next state
            state = np.random.choice(
                range(n_states),
                p=transition_matrix[state]
            )

        simulations.append(prices)

    return np.array(simulations)


def plot_simulations(sims):
    

    plt.figure(figsize=(12, 6))

    for i in range(min(30, sims.shape[0])):
        plt.plot(sims[i], alpha=0.4)

    plt.title("Regime-Aware Monte Carlo Simulations")
    plt.xlabel("Time Steps")
    plt.ylabel("Simulated Price")

    plt.show()

def compute_simulation_returns(sims):
    

    final_prices = sims[:, -1]

    initial_prices = sims[:, 0]

    returns = (final_prices / initial_prices) - 1

    return returns

def plot_return_distribution(returns):
    
    plt.figure(figsize=(10, 5))

    plt.hist(returns, bins=30)

    plt.title("Distribution of Simulated Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.show()

def compute_var(returns, alpha=0.05):
    
    var = np.percentile(returns, alpha * 100)

    return var

def compute_expected_shortfall(returns, alpha=0.05):
    

    var = compute_var(returns, alpha)

    es = returns[returns <= var].mean()

    return es


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    features_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    regimes_path = os.path.join(BASE_DIR, "data", "processed", "regimes.csv")

    df = load_data(features_path, regimes_path)

    params = estimate_regime_params(df)

    transition_matrix = estimate_transition_matrix(df["regime"])

    sims = simulate_paths(
        params,
        transition_matrix,
        start_price=100,
        n_steps=252,
        n_simulations=100
    )

    logging.info(f"Simulations shape: {sims.shape}")
    plot_simulations(sims)
    returns = compute_simulation_returns(sims)

    plot_return_distribution(returns)
    var_95 = compute_var(returns, alpha=0.05)

    es_95 = compute_expected_shortfall(returns, alpha=0.05)

    logging.info(f"VaR (95%): {var_95:.4f}")

    logging.info(f"Expected Shortfall (95%): {es_95:.4f}")
if __name__ == "__main__":
    main()