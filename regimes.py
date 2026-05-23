import os
import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_features(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading features from {file_path}")

    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

    logging.info(f"Feature data shape: {df.shape}")

    return df

def prepare_hmm_data(features: pd.DataFrame) -> pd.DataFrame:
    

    logging.info("Preparing HMM input data")

    log_returns = features["log_returns"]
    volatility = features["volatility"]

    
    df = pd.DataFrame({
        "log_returns": log_returns.mean(axis=1),
        "volatility": volatility.mean(axis=1)
    })

    df = df.dropna()

    logging.info(f"HMM input shape: {df.shape}")

    return df

def train_hmm(data: pd.DataFrame, n_states: int = 3) -> GaussianHMM:
    

    logging.info(f"Training HMM with {n_states} states")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    model.fit(data.values)

    logging.info("HMM training complete")

    return model

def predict_regimes(model: GaussianHMM, data: pd.DataFrame) -> pd.Series:
   

    logging.info("Predicting market regimes")

    states = model.predict(data.values)

    regimes = pd.Series(states, index=data.index, name="regime")

    return regimes

def save_regimes(regimes: pd.Series, output_path: str):
    logging.info(f"Saving regimes to {output_path}")

    regimes.to_csv(output_path)

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    input_path = os.path.join(BASE_DIR, "data", "processed", "features.csv")
    output_dir = os.path.join(BASE_DIR, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    features = load_features(input_path)

    hmm_data = prepare_hmm_data(features)

    model = train_hmm(hmm_data, n_states=3)

    regimes = predict_regimes(model, hmm_data)

    output_path = os.path.join(output_dir, "regimes.csv")
    save_regimes(regimes, output_path)

    logging.info("Regime detection completed")


if __name__ == "__main__":
    main()