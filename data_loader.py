import numpy as np
import pandas as pd
import yfinance as yf
import logging
import os
from config import ASSETS,START_DATE,END_DATE

logging.basicConfig(level=logging.INFO)

def data_download():
    logging.info("Downloading market data...")
    data=yf.download(
        tickers=ASSETS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )

    close_prices = data['Close']

    logging.info(f"Downloaded data shape: {close_prices.shape}")

    return close_prices

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "data", "raw")

    os.makedirs(SAVE_PATH, exist_ok=True)

    prices = data_download()

    file_path = os.path.join(SAVE_PATH, "market_data.csv")
    prices.to_csv(file_path)

    logging.info(f"Data saved to {file_path}")