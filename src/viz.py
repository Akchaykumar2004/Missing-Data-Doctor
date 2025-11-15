import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_bar(df: pd.DataFrame, out_path: str) -> None:
    missing_pct = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    missing_pct.plot(kind="bar")
    plt.ylabel("Missing %")
    plt.title("Missingness per feature")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def missing_heatmap(df: pd.DataFrame, out_path: str, max_rows: int = 200) -> None:
    sample = df.copy()
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)
    plt.figure(figsize=(10, 6))
    sns.heatmap(sample.isna(), cbar=False)
    plt.title("Missingness matrix (sampled rows)")
    plt.xlabel("Features")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
