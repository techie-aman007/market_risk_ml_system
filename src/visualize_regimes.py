import os
import logging

import pandas as pd
import matplotlib.pyplot as plt



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(prices_path: str, regimes_path: str) -> pd.DataFrame:
    logging.info("Loading data")

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    regimes = pd.read_csv(regimes_path, index_col=0, parse_dates=True)

    regimes.columns = ["regime"]

    df = prices.join(regimes, how="inner")

    logging.info(f"Combined data shape: {df.shape}")

    return df



def plot_regimes(df: pd.DataFrame, asset: str):
  
    if asset not in df.columns:
        raise ValueError(f"{asset} not found in dataset")

    logging.info(f"Plotting regimes for {asset}")

    fig, ax = plt.subplots(figsize=(14, 6))

    
    ax.plot(df.index, df[asset], label="Price")

    
    for regime in sorted(df["regime"].unique()):
        subset = df[df["regime"] == regime]

        ax.scatter(
            subset.index,
            subset[asset],
            label=f"Regime {regime}",
            s=10
        )

    ax.set_title(f"{asset} Price with Market Regimes")
    ax.legend()

    plt.show()


def analyze_regimes(df: pd.DataFrame) -> pd.DataFrame:
    

    logging.info("Analyzing regimes")

    
    price_df = df.drop(columns=["regime"])
    full_returns = price_df.pct_change()

    stats = []

    for regime in sorted(df["regime"].unique()):
        mask = df["regime"] == regime

        subset_returns = full_returns[mask]

        mean_return = subset_returns.mean().mean()
        volatility = subset_returns.std().mean()

        stats.append({
            "regime": regime,
            "mean_return": mean_return,
            "volatility": volatility,
            "count": mask.sum()
        })

    stats_df = pd.DataFrame(stats)

    logging.info("\n" + str(stats_df))

    return stats_df

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    prices_path = os.path.join(BASE_DIR, "data", "raw", "market_data.csv")
    regimes_path = os.path.join(BASE_DIR, "data", "processed", "regimes.csv")

    df = load_data(prices_path, regimes_path)

    # Choose a representative asset
    plot_regimes(df, asset=df.columns[0])

    analyze_regimes(df)


if __name__ == "__main__":
    main()