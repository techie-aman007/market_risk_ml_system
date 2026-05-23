import os 
import numpy as np
import pandas as pd
from typing import Tuple
import logging

from config import START_DATE, END_DATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(file_path : str) ->pd.DataFrame :

    logging.info(f"Loading data from {file_path}")

    df=pd.read_csv(file_path,index_col=0,parse_dates=True)
    df.sort_index(inplace=True)

    logging.info(f"Loaded data shape: {df.shape}")

    return df

def compute_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    logging.info("Computing returns")

    returns = df.pct_change()

    log_returns = np.log(df / df.shift(1))

    return returns, log_returns

def compute_volatility(log_returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    
    logging.info(f"Computing rolling volatility (window={window})")

    volatility = log_returns.rolling(window=window).std()

    return volatility

def compute_momentum(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    
    logging.info(f"Computing momentum (window={window})")

    momentum = df.pct_change(periods=window)

    return momentum


def compute_zscore(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    
    logging.info(f"Computing z-score (window={window})")

    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()

    zscore = (df - rolling_mean) / rolling_std

    return zscore

def build_feature_set(prices: pd.DataFrame) -> pd.DataFrame:
    
    logging.info("Building feature set")

    returns, log_returns = compute_returns(prices)
    volatility = compute_volatility(log_returns)
    momentum = compute_momentum(prices)
    zscore = compute_zscore(prices)

    
    features = pd.concat(
        {
            "returns": returns,
            "log_returns": log_returns,
            "volatility": volatility,
            "momentum": momentum,
            "zscore": zscore,
        },
        axis=1
    )

    logging.info(f"Feature set shape: {features.shape}")

    return features

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    input_path = os.path.join(BASE_DIR, "data", "raw", "market_data.csv")
    output_dir = os.path.join(BASE_DIR, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    prices = load_data(input_path)

    features = build_feature_set(prices)

    output_path = os.path.join(output_dir, "features.csv")
    features.to_csv(output_path)

    logging.info(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()

