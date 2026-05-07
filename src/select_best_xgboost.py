from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data_processed" / "onehot_dataset.csv"
MODELS_DIR = ROOT / "outputs" / "models"
OUTPUT_DIR = ROOT / "outputs"

SELECTION_PATH = OUTPUT_DIR / "best_model_selection.csv"
TEST_METRICS_PATH = OUTPUT_DIR / "best_model_test_metrics.json"


def load_data():
    """Load one-hot dataset and recreate the same 70/15/15 split."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label", "weight"])
    X.columns = (
        X.columns
        .astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "less_than", regex=False)
    )

    y = df["label"]
    w = df["weight"]

    X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
        X,
        y,
        w,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
        X_temp,
        y_temp,
        w_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print("\n--- Recreated data split ---")
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("\nValidation label distribution:")
    print(y_val.value_counts(normalize=True))
    print("\nTest label distribution:")
    print(y_test.value_counts(normalize=True))

    return X_val, X_test, y_val, y_test


def load_xgboost_model(model_path):
    """Load one saved XGBoost model."""
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def evaluate_thresholds(model, X_val, y_val, model_name):
    """Evaluate one model on validation data across thresholds."""
    y_prob = model.predict_proba(X_val)[:, 1]
    val_roc_auc = roc_auc_score(y_val, y_prob)

    rows = []

    # 0.05, 0.06, ..., 0.95
    thresholds = [i / 100 for i in range(5, 96)]

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "model_name": model_name,
            "threshold": threshold,
            "val_roc_auc": val_roc_auc,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_precision_1": precision_score(y_val, y_pred, zero_division=0),
            "val_recall_1": recall_score(y_val, y_pred, zero_division=0),
            "val_f1_1": f1_score(y_val, y_pred, zero_division=0),
        })

    return rows


def evaluate_test(model, X_test, y_test, threshold, model_name):
    """Evaluate selected model once on the test set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    test_metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "test_precision_1": precision_score(y_test, y_pred, zero_division=0),
        "test_recall_1": recall_score(y_test, y_pred, zero_division=0),
        "test_f1_1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            digits=4,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n--- Final Test Metrics ---")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print("Test Accuracy:", test_metrics["test_accuracy"])
    print("Test ROC AUC:", test_metrics["test_roc_auc"])
    print("Test Precision class 1:", test_metrics["test_precision_1"])
    print("Test Recall class 1:", test_metrics["test_recall_1"])
    print("Test F1 class 1:", test_metrics["test_f1_1"])

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nSaved test metrics to: {TEST_METRICS_PATH}")

    return test_metrics


def main():
    X_val, X_test, y_val, y_test = load_data()

    model_paths = sorted(MODELS_DIR.glob("*.json"))

    if not model_paths:
        raise FileNotFoundError(f"No model JSON files found in {MODELS_DIR}")

    print(f"\nFound {len(model_paths)} saved models.")

    all_rows = []

    for model_path in model_paths:
        model_name = model_path.stem
        print(f"Evaluating validation thresholds for: {model_name}")

        model = load_xgboost_model(model_path)
        rows = evaluate_thresholds(
            model=model,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
        )
        all_rows.extend(rows)

    selection_df = pd.DataFrame(all_rows)

    # Main rule:
    # 1. Prefer better validation ROC AUC.
    # 2. Then prefer better positive-class F1 after threshold tuning.
    selection_by_auc = selection_df.sort_values(
        by=["val_roc_auc", "val_f1_1"],
        ascending=[False, False],
    )

    # Alternative rule:
    # 1. Prefer better positive-class F1.
    # 2. Then prefer better validation ROC AUC.
    selection_by_f1 = selection_df.sort_values(
        by=["val_f1_1", "val_roc_auc"],
        ascending=[False, False],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection_df.to_csv(SELECTION_PATH, index=False)
    selection_by_auc.to_csv(OUTPUT_DIR / "best_model_selection_by_auc.csv", index=False)
    selection_by_f1.to_csv(OUTPUT_DIR / "best_model_selection_by_f1.csv", index=False)

    print(f"\nSaved all validation threshold results to: {SELECTION_PATH}")

    print("\n--- Top 10 by validation ROC AUC, then F1 ---")
    print(selection_by_auc.head(10))

    print("\n--- Top 10 by validation F1, then ROC AUC ---")
    print(selection_by_f1.head(10))

    # Final selection rule for test:
    # Use validation ROC AUC as the primary model-selection metric,
    # and use validation F1 to select the threshold.
    best_row = selection_by_auc.iloc[0]

    best_model_name = best_row["model_name"]
    best_threshold = float(best_row["threshold"])

    print("\n--- Selected Final Model ---")
    print(best_row)

    best_model_path = MODELS_DIR / f"{best_model_name}.json"
    best_model = load_xgboost_model(best_model_path)

    evaluate_test(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        threshold=best_threshold,
        model_name=best_model_name,
    )


if __name__ == "__main__":
    main()