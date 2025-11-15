import pandas as pd

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = df.shape[0]
    summary = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / total
        summary.append({
            "column": col,
            "missing_count": int(n_missing),
            "missing_pct": float(pct)
        })
    return pd.DataFrame(summary).sort_values("missing_pct", ascending=False)
