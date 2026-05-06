from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COLUMNS_PATH = ROOT / "data_raw" / "census-bureau.columns"
DATA_PATH = ROOT / "data_raw" / "census-bureau.data"

PROCESSED_DIR = ROOT / "data_processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_dataset.csv"

ONEHOT_DATA_PATH = PROCESSED_DIR / "onehot_dataset.csv"


def build_onehot_dataset(df):
    """Build final feature matrix with one-hot encoded categorical variables."""
    df = df.copy()

    target_col = "label"
    weight_col = "weight"

    binary_cols = [
        "year",
        "sex",
    ]

    numeric_cols = [
        "weeks worked in year",
        "dividends from stocks",
        "capital losses",
        "capital gains",
        "wage per hour",
        "age",
    ]

    categorical_cols = [
        "veterans benefits",
        "fill inc questionnaire for veteran's admin",
        "own business or self employed",
        "citizenship",
        "country of birth self",
        "country of birth mother",
        "country of birth father",
        "family members under 18",
        "num persons worked for employer",
        "migration prev res in sunbelt",
        "live in this house 1 year ago",
        "migration code-move within reg",
        "migration code-change in reg",
        "migration code-change in msa",
        "detailed household summary in household",
        "detailed household and family stat",
        "state of previous residence",
        "region of previous residence",
        "tax filer stat",
        "full or part time employment stat",
        "reason for unemployment",
        "member of a labor union",
        "hispanic origin",
        "race",
        "major occupation code",
        "major industry code",
        "marital stat",
        "enroll in edu inst last wk",
        "education",
        "detailed occupation recode",
        "detailed industry recode",
        "class of worker",
    ]

    # Safety check: make sure all requested columns exist.
    required_cols = binary_cols + numeric_cols + categorical_cols + [target_col, weight_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")
    

    # target and weight
    y = df[target_col].map({"- 50000.": 0,"50000+.": 1})
    w = df[weight_col]
    
    
    # Binary columns.
    print("\n--- Binary column value counts before encoding ---")
    for col in binary_cols:
        print(df[col].value_counts(dropna=False))

    df["sex"] = df["sex"].map({"Male": 1,"Female": 0})
    df["year"] = df["year"].map({94: 1,95: 0})


    #print("\n--- Binary columns after encoding ---")
    #print(df[binary_cols].head())
    #print(df[binary_cols].dtypes)
    #print(df[binary_cols].isna().sum())

    X_binary = df[binary_cols].astype(int)
    X_numeric = df[numeric_cols].apply(pd.to_numeric)
    X_cat = pd.get_dummies(df[categorical_cols],columns=categorical_cols, drop_first=False,dtype=int)

   

    #print(X_binary.head())
    #print(X_numeric.head())
    #print(X_cat.shape)


    #print("\n--- Feature block shapes ---")
    #print("X_binary:", X_binary.shape)
    #print("X_numeric:", X_numeric.shape)
    #print("X_cat:", X_cat.shape)


    # Combine all features
    X = pd.concat([X_binary, X_numeric, X_cat], axis=1)

    # Combine X + y + weight
    final_df = pd.concat(
        [
            X,
            y.rename(target_col),
            w.rename(weight_col),
        ],
        axis=1,
    )

    print("\n--- Final dataset ---")
    print("Shape:", final_df.shape)
    print("Last two columns:", final_df.columns[-2:].tolist())
    print(final_df.head())






    #print(df['sex'])
    #print(df['year'])


    #X_binary = df[binary_cols].astype(int)
    #X_numeric = df[numeric_cols].apply(pd.to_numeric)
    #print(X_binary)
    







    #X_cat = pd.get_dummies(df[categorical_cols],columns=categorical_cols, drop_first=False,dtype=int)

    #print(X_cat.shape)
    #print(X_cat.head())
    #print(X_cat.columns[:20].tolist())

    #X = pd.concat([X_binary, X_numeric, X_cat], axis=1)

    #final_df = pd.concat(
     #   [
      #      X,
       #     df[target_col].rename(target_col),
        #    df[weight_col].rename(weight_col),
        #],
        #axis=1,
    #)

    return final_df



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

    #####################
    # 2. one-hot encoding
    onehot_df = build_onehot_dataset(cleaned_df)
    #onehot_df = build_onehot_dataset(cleaned_df)

    onehot_df.to_csv(ONEHOT_DATA_PATH, index=False)
    print(f"\nSaved one-hot dataset to: {ONEHOT_DATA_PATH}")


if __name__ == "__main__":
    main()