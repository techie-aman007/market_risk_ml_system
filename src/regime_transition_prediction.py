import os
import logging

import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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
def load_data(features_path: str, regimes_path: str):

    logging.info("Loading datasets")

    features = pd.read_csv(
        features_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    regimes = pd.read_csv(
        regimes_path,
        index_col=0,
        parse_dates=True
    )

    regimes.columns = ["regime"]

    # -----------------------------------------------------
    # Flatten columns
    # -----------------------------------------------------
    features.columns = [
        f"{col[0]}_{col[1]}"
        for col in features.columns
    ]

    df = features.join(regimes, how="inner")

    logging.info(f"Dataset shape: {df.shape}")

    return df


# =========================================================
# Create Transition Target
# =========================================================
def create_transition_target(
    df: pd.DataFrame,
    horizon: int = 5
):
    """
    Create target:
    1 if regime changes within next N days.
    """

    logging.info(
        f"Creating {horizon}-day transition target"
    )

    regimes = df["regime"]

    targets = []

    # -----------------------------------------------------
    # Create future transition labels
    # -----------------------------------------------------
    for i in range(len(regimes)):

        current_regime = regimes.iloc[i]

        future_window = regimes.iloc[
            i + 1:i + 1 + horizon
        ]

        # Transition occurs if any future regime differs
        transition = (
            future_window != current_regime
        ).any()

        targets.append(int(transition))

    df["target"] = targets

    # Remove incomplete future horizon rows
    df = df.iloc[:-horizon]

    logging.info(
        f"Transition rate: "
        f"{df['target'].mean():.4f}"
    )

    return df

# =========================================================
# Prepare Dataset
# =========================================================
def prepare_dataset(df: pd.DataFrame):

    logging.info("Preparing dataset")

    X = df.drop(columns=["regime", "target"])

    y = df["target"]

    # -----------------------------------------------------
    # Remove sparse columns
    # -----------------------------------------------------
    nan_fraction = X.isna().mean()

    cols_to_keep = nan_fraction[
        nan_fraction < 0.3
    ].index

    X = X[cols_to_keep]

    # -----------------------------------------------------
    # Fill missing values
    # -----------------------------------------------------
    X = X.ffill()

    # Remove remaining NaNs
    valid_idx = X.dropna().index

    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")

    return X, y


# =========================================================
# Split Data
# =========================================================
def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# Train Model
# =========================================================
def train_model(X_train, y_train):

    logging.info(
        "Training XGBoost transition model"
    )

    # -----------------------------------------------------
    # Handle imbalance
    # -----------------------------------------------------
    positive_ratio = y_train.mean()

    scale_pos_weight = (
        (1 - positive_ratio) / positive_ratio
    )

    logging.info(
        f"Scale positive weight: "
        f"{scale_pos_weight:.2f}"
    )

    # -----------------------------------------------------
    # XGBoost model
    # -----------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# Evaluate Model
# =========================================================
def evaluate_model(model, X_test, y_test):

    logging.info("Evaluating probabilistic model")

    # -----------------------------------------------------
    # Hard predictions
    # -----------------------------------------------------
    predictions = model.predict(X_test)

    # -----------------------------------------------------
    # Transition probabilities
    # -----------------------------------------------------
    probabilities = model.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    logging.info(
        f"Accuracy: {accuracy:.4f}"
    )

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    logging.info(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # Precision-Recall AUC
    # -----------------------------------------------------
    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = auc(recall, precision)

    logging.info(
        f"PR-AUC: {pr_auc:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------
    logging.info(
        "\nClassification Report:\n"
    )

    logging.info(
        "\n" + classification_report(
            y_test,
            predictions
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    logging.info(
        "\nConfusion Matrix:\n"
    )

    logging.info(
        "\n" + str(
            confusion_matrix(
                y_test,
                predictions
            )
        )
    )

    return probabilities

def plot_transition_probabilities(probabilities):

    plt.figure(figsize=(12, 5))

    plt.plot(probabilities)

    plt.title(
        "Predicted Regime Transition Probabilities"
    )

    plt.xlabel("Time")

    plt.ylabel("Transition Probability")

    plt.show()

# =========================================================
# Feature Importance
# =========================================================
def feature_importance(model, X):

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(
        ascending=False
    )

    logging.info(
        "\nTop 15 Features:\n"
    )

    logging.info(
        "\n" + str(importance.head(15))
    )

def shap_analysis(model, X):

    logging.info(
        "Running SHAP explainability analysis"
    )

    # -----------------------------------------------------
    # Create explainer
    # -----------------------------------------------------
    explainer = shap.TreeExplainer(model)

    # -----------------------------------------------------
    # Sample subset for speed
    # -----------------------------------------------------
    X_sample = X.sample(
        min(500, len(X)),
        random_state=42
    )

    # -----------------------------------------------------
    # Compute SHAP values
    # -----------------------------------------------------
    shap_values = explainer.shap_values(X_sample)

    # -----------------------------------------------------
    # Summary plot
    # -----------------------------------------------------
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.title(
        "SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# Persistence Baseline
# =========================================================
def persistence_baseline(y):

    logging.info(
        "Running persistence baseline"
    )

    # Always predict NO transition
    predictions = [0] * len(y)

    accuracy = (
        predictions == y.values
    ).mean()

    logging.info(
        f"Baseline Accuracy: "
        f"{accuracy:.4f}"
    )

    return accuracy


# =========================================================
# Main
# =========================================================
def main():

    BASE_DIR = os.path.dirname(
        os.path.dirname(__file__)
    )

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
    # Create transition labels
    # -----------------------------------------------------
    df = create_transition_target(df)

    # -----------------------------------------------------
    # Prepare dataset
    # -----------------------------------------------------
    X, y = prepare_dataset(df)

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------
    persistence_baseline(y)

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    model = train_model(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------
    probabilities = evaluate_model(
    model,
    X_test,
    y_test
    )
    # -----------------------------------------------------
# Save probabilities
# -----------------------------------------------------
    BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
    )

    output_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "transition_probabilities.csv"
    )

    pd.DataFrame(
    probabilities,
    index=X_test.index,
    columns=["transition_probability"]
    ).to_csv(output_path)

    logging.info(
    "Saved transition probabilities"
     )
    plot_transition_probabilities(
    probabilities
    )
    evaluate_custom_threshold(
    probabilities,
    y_test,
    threshold=0.2
    )

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    feature_importance(
        model,
        X
    )

    shap_analysis(
    model,
    X
    )
          
def evaluate_custom_threshold(
    probabilities,
    y_test,
    threshold=0.2
):

    logging.info(
        f"Evaluating threshold = {threshold}"
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    logging.info(
        "\n" + classification_report(
            y_test,
            predictions
        )
    )


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    main()