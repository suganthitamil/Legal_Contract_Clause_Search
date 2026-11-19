from pathlib import Path

import pandas as py


def main():
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "cuad" / "CUAD_v1" / "master_clauses.csv"
    # Load the CSV file into a DataFrame
    df = py.read_csv(csv_path)
    # Display the first few rows of the DataFrame
    print(df.shape[0])
    print(df["Agreement Date-Answer"].shape[0])
    print(df["Agreement Date-Answer"].isna().sum())
    print((df["Agreement Date"].astype(str).str.strip() == "[]").sum())