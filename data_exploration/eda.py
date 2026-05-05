from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

################################
# EDA display toggles
SHOW_PATH_CHECKS = False  # Print input/output paths and file existence checks.
SHOW_RAW_PREVIEW = False # Print column names and a few raw data rows for schema inspection.
#SHOW_LOADED_DATA_PREVIEW = True  # Print DataFrame shape and first few loaded rows.

###############################
# Define Paths
ROOT = Path(__file__).resolve().parents[1]

# Input paths
COLUMNS_PATH = ROOT / "data_raw" / "census-bureau.columns"
DATA_PATH = ROOT / "data_raw" / "census-bureau.data"

# Output paths
FIGURE_DIR = ROOT / "outputs" / "figures" / "eda"
PROCESSED_DIR = ROOT / "data_processed"

################################
def setup_directories():
    """Create output directories used by the EDA script."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def print_path_checks():
    """Print resolved file paths and check whether raw input files exist."""
    print("- Columns path:\n", COLUMNS_PATH)
    print("- Data path:\n", DATA_PATH)
    print("- Figure directory:\n", FIGURE_DIR)
    print("- Processed data directory:\n", PROCESSED_DIR)

    print("- Columns file exists:", COLUMNS_PATH.exists())
    print("- Data file exists:", DATA_PATH.exists())


def inspect_raw_files(first_n=3):
    """Print raw column names and the first few raw data rows."""
    print("\n--- Column names: ---")
    with open(COLUMNS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        column_lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(column_lines):
        print(f"{i}: {line}")

    print(f"\nNumber of non-empty lines in columns file: {len(column_lines)}")

    print(f"\n--- First {first_n} lines of data file ---")
    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= first_n:
                break
            print(f"{i}: {line.rstrip()}")

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

def summarize_columns(df):
    """Create a summary table for all columns."""
    summary_rows = []

    for col in df.columns:
        s = df[col]
        summary_rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": s.nunique(dropna=False),
            "n_missing": s.isna().sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = ROOT/"data_exploration"/ "column_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Column Summary ---")
    print(summary_df[["column", "dtype", "n_unique", "n_missing"]])
    print(f"\nSaved column summary to: {summary_path}")

    return summary_df


def clean_filename(name):
    """Convert a column name into a simple figure filename."""
    return str(name).strip().replace(" ", "_").replace("'", "")

def plot_one_feature(df, col, bins="auto"):
    """Plot one feature and save the figure."""
    s = df[col]
    save_name = clean_filename(col)

    n_missing = s.isna().sum()
    missing_pct = n_missing / len(s) * 100

    plt.figure(figsize=(10, 5))

    if pd.api.types.is_numeric_dtype(s):
        non_missing = s.dropna()

        # Use matplotlib directly because pandas hist may not handle bins="auto" reliably.
        plt.hist(non_missing, bins=bins, edgecolor="black")

        min_val = non_missing.min()
        max_val = non_missing.max()
        plt.grid()

        plt.title(
            f"Histogram: {col}\n"
            f"missing={n_missing} ({missing_pct:.2f}%), "
            f"min={min_val:g}, max={max_val:g}"
        )
        save_path = FIGURE_DIR / f"{save_name}_histogram.png"

    else:
        value_counts = s.value_counts(dropna=False)
        value_counts.index = [
            "Missing" if pd.isna(x) else str(x)
            for x in value_counts.index
        ]

        value_counts.plot(kind="bar")
        plt.grid()

        plt.title(
            f"Value Counts: {col}\n"
            f"missing={n_missing} ({missing_pct:.2f}%), "
            f"n_unique={s.nunique(dropna=False)}"
        )
        plt.xticks(rotation=45, ha="right")
        save_path = FIGURE_DIR / f"{save_name}_value_counts.png"

    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {save_path}")


def inspect_one_feature(df, col):
    """Print basic EDA information for one selected feature."""
    print("\n" + "=" * 80)
    print(f"Feature: {col}")
    print(f"dtype: {df[col].dtype}")
    print(f"n_unique: {df[col].nunique(dropna=False)}")
    print(f"n_missing: {df[col].isna().sum()}")

    print("\n--- Value counts ---")
    # Do not sort here because categorical columns may mix strings and NaN.
    print(df[col].value_counts(dropna=False))

    if pd.api.types.is_numeric_dtype(df[col]):
        print("\n--- Numeric summary ---")
        print(df[col].describe())

    plot_one_feature(df, col)


def weighted_value_counts(df, col, weight_col="weight"):
    """Compute weighted counts and percentages for a categorical column."""
    out = (
        df.groupby(col, dropna=False)[weight_col]
        .sum()
        .reset_index(name="weighted_count")
    )
    out["weighted_pct"] = out["weighted_count"] / out["weighted_count"].sum()
    return out.sort_values("weighted_count", ascending=False)


def compare_weighted_unweighted_counts(df, col, weight_col="weight"):
    """Compare raw record counts with survey-weighted counts for a categorical column."""
    raw = (
        df[col]
        .value_counts(dropna=False)
        .reset_index()
    )
    raw.columns = [col, "raw_count"]
    raw["raw_pct"] = raw["raw_count"] / raw["raw_count"].sum()

    weighted = weighted_value_counts(df, col, weight_col)

    out = raw.merge(weighted, on=col, how="outer")
    out = out.sort_values("weighted_count", ascending=False)

    print(f"\n--- Weighted vs unweighted: {col} ---")
    print(out)

    return out


def main():
    setup_directories()

    if SHOW_PATH_CHECKS:
        print_path_checks()

    if SHOW_RAW_PREVIEW:
        inspect_raw_files()

    column_names = load_column_names()
    print("\n--- Loaded column names ---")
    print(f"Number of columns: {len(column_names)}")
    
    # Load raw census data using the schema above.
    df = load_raw_data(column_names)
    print("\n--- Loaded DataFrame ---")
    print("Shape:", df.shape)

    print("\n--- First 5 rows ---")
    print(df.head())

    summary_df = summarize_columns(df)

    # inspect one feature

    feature_name=column_names[41]
    inspect_one_feature(df, feature_name)
    #df_nonzero = df[df[feature_name] != 0]
    #inspect_one_feature(df_nonzero, feature_name)

    for c in column_names:
        compare_weighted_unweighted_counts(df, c)
    #compare_weighted_unweighted_counts(df, "education")
    #compare_weighted_unweighted_counts(df, "class of worker")
    #compare_weighted_unweighted_counts(df, "sex")
    




    


if __name__ == "__main__":
    main()