import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def evaluate_imputations(df: pd.DataFrame, target: str, task: str, imputed_versions: dict) -> dict:
    results: dict = {}
    for name, df_imp in imputed_versions.items():
        if target not in df_imp.columns:
            continue
        X = df_imp.drop(columns=[target])
        y = df_imp[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if task == "classification":
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
            auc = roc_auc_score(y_test, proba)
            acc = accuracy_score(y_test, pred)
            results[name] = {"AUC": float(auc), "Accuracy": float(acc)}
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            rmse = mean_squared_error(y_test, pred, squared=False)
            results[name] = {"R2": float(r2), "RMSE": float(rmse)}
    return results
