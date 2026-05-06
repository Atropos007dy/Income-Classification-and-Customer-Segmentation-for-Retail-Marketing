from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COLUMNS_PATH = ROOT / "data_raw" / "census-bureau.columns"
DATA_PATH = ROOT / "data_raw" / "census-bureau.data"

PROCESSED_DIR = ROOT / "data_processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_dataset.csv"


def load_column_names():
    """Load column names from the provided columns file."""
    with open(COLUMNS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        column_names = [line.strip() for line in f if line.strip()]
    return column_names


def load_raw_data(column_names):
    """Load the raw comma-delimited data file into a pandas DataFrame."""
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=column_names,
        sep=",",
        skipinitialspace=True,
    )
    return df


def clean_raw_data(df):
    """Handling missing values."""
    df = df.copy()

    string_cols = df.select_dtypes(include=["object", "string"]).columns

    # Remove accidental leading/trailing spaces.
    for col in string_cols:
        df[col] = df[col].str.strip()

    # Recode string-coded unknown values and true categorical missing values.
    df[string_cols] = df[string_cols].replace("?", "Unknown")
    df[string_cols] = df[string_cols].fillna("Unknown")

    return df


def inspect_one_feature(df, col):
    """Print basic information for one selected feature."""
    print("\n" + "=" * 80)
    print(f"Feature: {col}")
    print(f"dtype: {df[col].dtype}")
    print(f"n_unique: {df[col].nunique(dropna=False)}")
    print(f"n_missing: {df[col].isna().sum()}")

    print("\n--- Value counts ---")
    print(df[col].value_counts(dropna=False).head(20))

def main():
    ###################
    # 1. missing value handling
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    column_names = load_column_names()
    df = load_raw_data(column_names)

    #print("\n--- Raw dataset ---")
    #print("Shape:", df.shape)
    #print("Total missing values:", df.isna().sum().sum())

    cleaned_df = clean_raw_data(df)

    #print("\n--- Cleaned dataset ---")
    #print("Shape:", cleaned_df.shape)
    #print("Total missing values:", cleaned_df.isna().sum().sum())

    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {CLEANED_DATA_PATH}")

    #for c in column_names:
    #    inspect_one_feature(cleaned_df, c)


if __name__ == "__main__":
    main()