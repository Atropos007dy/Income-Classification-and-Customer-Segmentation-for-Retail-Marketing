from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split,ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report,precision_score,recall_score,f1_score

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data_processed" / "onehot_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
#MODEL_PATH = OUTPUT_DIR /"models"/"xgboost_baseline.json"

def tune_threshold(model, X_val, y_val, model_name):
    """Find the best classification threshold based on validation F1 for class 1."""
    y_prob = model.predict_proba(X_val)[:, 1]

    thresholds = [i / 100 for i in range(5, 96, 1)]
    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        results.append({
            "model_name": model_name,
            "threshold": threshold,
            "precision_1": precision,
            "recall_1": recall,
            "f1_1": f1,
        })

    threshold_df = pd.DataFrame(results)
    threshold_df = threshold_df.sort_values(by="f1_1", ascending=False)

    print(f"\n--- Threshold tuning: {model_name} ---")
    print(threshold_df.head(10))

    threshold_path = OUTPUT_DIR / "thresholds" / f"{model_name}_threshold_tuning.csv"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(threshold_path, index=False)

    print(f"Saved threshold tuning to: {threshold_path}")

    best_threshold = threshold_df.iloc[0]["threshold"]
    best_precision = threshold_df.iloc[0]["precision_1"]
    best_recall = threshold_df.iloc[0]["recall_1"]
    best_f1 = threshold_df.iloc[0]["f1_1"]

    return best_threshold, best_precision, best_recall, best_f1

def evaluate_model(model, X_eval, y_eval, split_name, model_name):
    '''
    Evaluate a trained classifier on a given dataset split and save metrics.
    '''
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_prob)

    report_dict = classification_report(
        y_eval,
        y_pred,
        digits=4,
        output_dict=True,
    )

    print(f"\n--- {split_name} Metrics: {model_name} ---")
    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    print(classification_report(y_eval, y_pred, digits=4))

    metrics = {
        "model_name": model_name,
        "split": split_name,
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": report_dict,
    }

    metrics_path = OUTPUT_DIR / "metrics" / f"{model_name}_{split_name.lower()}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {metrics_path}")

    return metrics


def train_xgboost_model(model_name, params, X_train, y_train, w_train, X_val, y_val):
    """Train an XGBoost model, evaluate on train and validation sets, and save model."""
    print("\n" + "=" * 80)
    print(f"Training model: {model_name}")
    print(f"Params: {params}")
    print("=" * 80)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        #scale_pos_weight=pos_weight,
        **params,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
    )

    train_metrics = evaluate_model(
        model=model,
        X_eval=X_train,
        y_eval=y_train,
        split_name="train",
        model_name=model_name,
    )

    val_metrics = evaluate_model(
        model=model,
        X_eval=X_val,
        y_eval=y_val,
        split_name="val",
        model_name=model_name,
    )
    best_threshold, best_precision, best_recall, best_f1 = tune_threshold(
        model=model,
        X_val=X_val,
        y_val=y_val,
        model_name=model_name,
    )

    model_path = OUTPUT_DIR / "models" / f"{model_name}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"Saved model to: {model_path}")

    return model, train_metrics, val_metrics, best_threshold, best_precision, best_recall, best_f1



def main():
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

    print("X shape:", X.shape)
    print("y distribution:")
    print(y.value_counts(normalize=True))
    print("weight summary:")
    print(w.describe())


    # split data, 70:15:15
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
    
    print("\n--- Split shapes ---")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    
    print("\n--- Label distributions ---")
    print("Train:")
    print(y_train.value_counts(normalize=True))
    
    print("\nValidation:")
    print(y_val.value_counts(normalize=True))
    
    print("\nTest:")
    print(y_test.value_counts(normalize=True))


    #define model and train
    #define a small hyperparameter search grid.

   # param_grid = {
    #"n_estimators": [200, 300, 500, 700],
    #"max_depth": [4, 5, 6, 7, 8],
    #"learning_rate": [0.03, 0.05,0.1],
    #"subsample": [0.8],
    #"colsample_bytree": [0.8],
    #"scale_pos_weight": [1, 5, 10, 15],
#}
    
    param_grid = {
    "n_estimators": [700, 900, 1100],
    "max_depth": [5, 6, 7, 8],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "scale_pos_weight": [1, 5, 10, 15],
}

    param_list = list(ParameterGrid(param_grid))
    print(f"Number of parameter combinations: {len(param_list)}")



    results = []

    for i, params in enumerate(param_list, start=1):
        model_name = f"xgboost_model_{240+i}"

        model, train_metrics, val_metrics, best_threshold, best_precision, best_recall, best_f1 = train_xgboost_model(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_val=X_val,
            y_val=y_val,
        )

        results.append({
            "model_name": model_name,
            **params,
            "train_accuracy": train_metrics["accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_precision_1": train_metrics["classification_report"]["1"]["precision"],
            "train_recall_1": train_metrics["classification_report"]["1"]["recall"],
            "train_f1_1": train_metrics["classification_report"]["1"]["f1-score"],
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_precision_1_at_0.5": val_metrics["classification_report"]["1"]["precision"],
            "val_recall_1_at_0.5": val_metrics["classification_report"]["1"]["recall"],
            "val_f1_1_at_0.5": val_metrics["classification_report"]["1"]["f1-score"],
            "best_threshold": best_threshold,
            "val_precision_1": best_precision,
            "val_recall_1": best_recall,
            "val_f1_1": best_f1,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=[ "val_roc_auc","val_f1_1"],ascending=[False, False])

    print("\n--- Model comparison ---")
    print(results_df)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(comparison_path, index=False)

    print(f"\nSaved model comparison to: {comparison_path}")
    print("\nBest model by validation F1 for class 1:")
    print(results_df.iloc[0])


if __name__ == "__main__":
    main()