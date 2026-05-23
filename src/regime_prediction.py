import os
import logging

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(features_path: str, regimes_path: str) -> pd.DataFrame:
    """
    Load feature dataset and regime labels.
    """

    logging.info("Loading datasets")

    # Load engineered features
    features = pd.read_csv(
        features_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    # Load regimes
    regimes = pd.read_csv(
        regimes_path,
        index_col=0,
        parse_dates=True
    )

    regimes.columns = ["regime"]

    # -----------------------------------------------------
    # Flatten MultiIndex columns
    # Example:
    # ('log_returns', 'SPY')
    # -> log_returns_SPY
    # -----------------------------------------------------
    features.columns = [
        f"{col[0]}_{col[1]}"
        for col in features.columns
    ]

    # Merge datasets
    df = features.join(regimes, how="inner")

    logging.info(f"Combined dataset shape: {df.shape}")

    return df


# =========================================================
# Prepare ML Dataset
# =========================================================
def prepare_dataset(df: pd.DataFrame):

    logging.info("Preparing supervised learning dataset")

    # -----------------------------------------------------
    # Features
    # -----------------------------------------------------
    X = df.drop(columns=["regime"])

    # -----------------------------------------------------
    # Target = next regime
    # -----------------------------------------------------
    y = df["regime"].shift(-1)

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------
    dataset = X.copy()
    dataset["target"] = y

    logging.info(f"Initial dataset shape: {dataset.shape}")

    # -----------------------------------------------------
    # Remove columns with too many NaNs
    # -----------------------------------------------------
    nan_fraction = dataset.isna().mean()

    cols_to_keep = nan_fraction[nan_fraction < 0.3].index

    dataset = dataset[cols_to_keep]

    logging.info(f"Shape after dropping sparse columns: {dataset.shape}")

    # -----------------------------------------------------
    # Forward fill remaining NaNs
    # -----------------------------------------------------
    dataset = dataset.ffill()

    # -----------------------------------------------------
    # Drop remaining NaNs
    # -----------------------------------------------------
    dataset = dataset.dropna()

    logging.info(f"Final cleaned dataset shape: {dataset.shape}")

    # -----------------------------------------------------
    # Split features/target
    # -----------------------------------------------------
    X = dataset.drop(columns=["target"])

    y = dataset["target"]

    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target vector shape: {y.shape}")

    return X, y


# =========================================================
# Train/Test Split
# =========================================================
def split_data(X, y):
    """
    Split time series dataset.

    IMPORTANT:
    shuffle=False preserves chronological order.
    """

    logging.info("Splitting train/test data")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    logging.info(f"Train size: {X_train.shape[0]}")
    logging.info(f"Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# =========================================================
# Train Model
# =========================================================
def train_model(X_train, y_train):
    """
    Train Random Forest classifier.
    """

    logging.info("Training Random Forest classifier")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    logging.info("Model training completed")

    return model


def walk_forward_validation(X, y):
    """
    Perform walk-forward validation.

    Train on expanding window
    and evaluate sequentially.
    """

    logging.info("Starting walk-forward validation")

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    train_size = int(len(X) * 0.6)

    step_size = int(len(X) * 0.1)

    accuracies = []

    start = train_size

    # -----------------------------------------------------
    # Walk-forward loop
    # -----------------------------------------------------
    while start + step_size < len(X):

        # Expanding training window
        X_train = X.iloc[:start]
        y_train = y.iloc[:start]

        # Next testing window
        X_test = X.iloc[start:start + step_size]
        y_test = y.iloc[start:start + step_size]

        logging.info(
            f"Train: {X_train.shape[0]} | "
            f"Test: {X_test.shape[0]}"
        )

        # -------------------------------------------------
        # Train model
        # -------------------------------------------------
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------
        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        accuracies.append(accuracy)

        logging.info(
            f"Window Accuracy: {accuracy:.4f}"
        )

        # Move forward
        start += step_size

    # -----------------------------------------------------
    # Final statistics
    # -----------------------------------------------------
    avg_accuracy = sum(accuracies) / len(accuracies)

    logging.info(
        f"\nAverage Walk-Forward Accuracy: "
        f"{avg_accuracy:.4f}"
    )

    return accuracies


def persistence_baseline(y):
    """
    Baseline:
    Predict next regime = current regime.
    """

    logging.info("Running persistence baseline")

    y_true = y.iloc[1:]

    y_pred = y.shift(1).iloc[1:]

    accuracy = (y_true == y_pred).mean()

    logging.info(
        f"Persistence Baseline Accuracy: "
        f"{accuracy:.4f}"
    )

    return accuracy
# =========================================================
# Evaluate Model
# =========================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification performance.
    """

    logging.info("Evaluating model")

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions
    )

    logging.info("\nClassification Report:\n")
    logging.info("\n" + report)


# =========================================================
# Feature Importance
# =========================================================
def analyze_feature_importance(model, X):
    """
    Analyze feature importance.
    """

    logging.info("Analyzing feature importance")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(
        ascending=False
    )

    logging.info("\nTop 15 Most Important Features:\n")
    logging.info("\n" + str(importance.head(15)))

    return importance


# =========================================================
# Main Pipeline
# =========================================================
def main():

    # -----------------------------------------------------
    # Base directory
    # -----------------------------------------------------
    BASE_DIR = os.path.dirname(
        os.path.dirname(__file__)
    )

    # -----------------------------------------------------
    # File paths
    # -----------------------------------------------------
    features_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "features.csv"
    )

    regimes_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "regimes.csv"
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    df = load_data(
        features_path,
        regimes_path
    )

    # -----------------------------------------------------
    # Prepare dataset
    # -----------------------------------------------------
    X, y = prepare_dataset(df)

    # -----------------------------------------------------
    # Train/Test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------
    model = train_model(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Evaluate model
    # -----------------------------------------------------
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    analyze_feature_importance(
        model,
        X
    )
    walk_forward_validation(X, y)
    persistence_baseline(y)

# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()