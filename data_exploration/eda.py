from pathlib import Path
import pandas as pd

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


    


if __name__ == "__main__":
    main()