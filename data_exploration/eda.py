from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Input paths
columns_path = ROOT / "data_raw" / "census-bureau.columns"
data_path = ROOT / "data_raw" / "census-bureau.data"

# Output paths
output_dir = ROOT / "outputs"
figure_dir = output_dir / "figures" / "eda"
processed_dir = ROOT / "data_processed"

figure_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

# Print paths and check files
print("- Columns path:\n", columns_path)
print("- Data path:\n", data_path)
print("- Figure directory:\n", figure_dir)
print("- Processed data directory:\n", processed_dir)

print("Columns file exists: ", columns_path.exists())
print("Data file exists: ", data_path.exists())

