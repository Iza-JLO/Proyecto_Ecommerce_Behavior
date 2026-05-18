from pathlib import Path
import json

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

from project_paths import DATA_DIR, REPORTS_DIR, ensure_project_dirs


def main():
    ensure_project_dirs()

    # Set the path to the file you'd like to load
    file_path = "orders.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "meruvakodandasuraj/e-commerce-customer-behavior-and-sales-20202026",
      file_path,
      # Provide any additional arguments like
      # sql_query or pandas_kwargs. See the
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print("First 5 records:")
    print(df.head())
    print()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    output_path = DATA_DIR / "orders.csv"
    df.to_csv(output_path, index=False)

    profile = {
        "dataset": "meruvakodandasuraj/e-commerce-customer-behavior-and-sales-20202026",
        "file": file_path,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(df.columns),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "output_path": str(output_path),
    }

    with (REPORTS_DIR / "raw_data_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, ensure_ascii=False)

    print(f"Saved dataset to: {output_path}")
    print(f"Saved profile to: {REPORTS_DIR / 'raw_data_profile.json'}")


if __name__ == "__main__":
    main()