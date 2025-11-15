import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def simple_impute(df: pd.DataFrame, strategy_num: str = "median", strategy_cat: str = "most_frequent") -> pd.DataFrame:
    df_num = df.select_dtypes(include=["number"])
    df_cat = df.select_dtypes(exclude=["number"])
    imp_num = SimpleImputer(strategy=strategy_num)
    imp_cat = SimpleImputer(strategy=strategy_cat)
    df_num_imp = pd.DataFrame(imp_num.fit_transform(df_num), columns=df_num.columns, index=df.index)
    if not df_cat.empty:
        df_cat_imp = pd.DataFrame(imp_cat.fit_transform(df_cat), columns=df_cat.columns, index=df.index)
    else:
        df_cat_imp = df_cat
    return pd.concat([df_num_imp, df_cat_imp], axis=1)[df.columns]

def knn_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df_num = df.select_dtypes(include=["number"])
    df_cat = df.select_dtypes(exclude=["number"])
    imp = KNNImputer(n_neighbors=n_neighbors)
    df_num_imp = pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns, index=df.index)
    if not df_cat.empty:
        df_cat_imp = df_cat.fillna(df_cat.mode().iloc[0])
    else:
        df_cat_imp = df_cat
    return pd.concat([df_num_imp, df_cat_imp], axis=1)[df.columns]

def iterative_impute(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes(include=["number"])
    imp = IterativeImputer(random_state=42)
    df_num_imp = pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns, index=df.index)
    df_cat = df.select_dtypes(exclude=["number"])
    if not df_cat.empty:
        df_cat_imp = df_cat.fillna(df_cat.mode().iloc[0])
    else:
        df_cat_imp = df_cat
    return pd.concat([df_num_imp, df_cat_imp], axis=1)[df.columns]
