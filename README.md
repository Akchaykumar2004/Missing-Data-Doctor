# Missing Data Doctor

A diagnostic & treatment suite for **missing values in tabular machine learning datasets**.

Missing Data Doctor helps you:

- **Quantify** how much data is missing and where  
- **Visualize** missingness patterns across features and rows  
- **Impute** missing values using multiple strategies  
- **Evaluate** how each imputation choice affects model performance  
- **Report** everything in a portable, self-contained HTML report

It is designed as a **practical data-science tool** you can drop into real workflows or showcase as a professional project on GitHub.

---

## Project Structure

```text
missing-data-doctor/
├── src/
│   ├── cli.py            # Main CLI entrypoint
│   ├── loaders.py        # CSV loading & schema helpers
│   ├── profiling.py      # Missingness summary and stats
│   ├── imputers.py       # Imputation strategies (simple, KNN, iterative)
│   ├── impact.py         # Downstream model impact analysis
│   ├── viz.py            # Plotting utilities for missing data
│   └── report.py         # Jinja2 HTML report generation
│
├── templates/
│   └── report.html       # HTML report template (embeds plots & tables)
│
├── data/
│   └── example_with_missing.csv   # Example dataset with missing values
│
├── outputs/
│   └── runs/
│       └── demo/         # Example run (created after you run the demo)
│           ├── missing_data_doctor.html
│           └── plots/
│               ├── missing_bar.png
│               └── missing_heatmap.png
│
├── reports/              # Optional alternative report location (if you use --report)
└── README.md
````

> **Flat layout**: all Python modules live directly under `src/` (no package folders, no `mdd`).

---

## Quickstart

### 1. Create and activate a virtual environment (Windows CMD)

```cmd
cd C:\Users\Amir\Desktop\missing-data-doctor
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```cmd
pip install pandas numpy scikit-learn matplotlib seaborn jinja2
```

### 3. Run the demo pipeline

This command:

* Loads `data/example_with_missing.csv`
* Profiles missingness
* Generates plots
* Runs 3 imputation strategies
* Evaluates a model for each
* Writes a **self-contained run folder** with plots + JSON + HTML

```cmd
python src\cli.py ^
  --data data\example_with_missing.csv ^
  --target target ^
  --task classification ^
  --out_dir outputs\runs\demo
```

You’ll get:

```text
outputs/runs/demo/
├── missing_summary.csv
├── impact.json          # model metrics per imputation strategy (if target provided)
├── summary.json         # combined summary (missingness + impact)
├── plots/
│   ├── missing_bar.png
│   └── missing_heatmap.png
└── missing_data_doctor.html
```

### 4. Open the HTML report

```cmd
start "" outputs\runs\demo\missing_data_doctor.html
```

---

## What the Example Dataset Looks Like

`data/example_with_missing.csv` is a small synthetic dataset:

```text
age | income | visits | score | target
25  | 30000  |   5    |  620  |   0
40  |        |   10   |  680  |   1
35  | 45000  |        |  640  |   0
    | 70000  |   12   |  720  |   1
28  | 34000  |   6    |       |   0
46  | 66000  |   11   |  700  |   1
31  |        |   7    |  630  |   0
54  | 75000  |   13   |  730  |   1
29  | 35000  |        |  615  |   0
43  | 59000  |   9    |  690  |   1
```

Key properties:

* **10 rows** with 5 columns: `age`, `income`, `visits`, `score`, `target`
* Missing values:

  * `income`: 2 missing → **20%**
  * `visits`: 2 missing → **20%**
  * `age`: 1 missing → **10%**
  * `score`: 1 missing → **10%**
  * `target`: no missing
* `target` is a binary label: `0/1` (classification problem)

This toy dataset is intentionally small so you can **easily interpret the plots and metrics** created by Missing Data Doctor.

---

## Generated Figures (and How to Read Them)

After running the demo, the key figures live here:

```text
outputs/runs/demo/plots/
  ├── missing_bar.png
  └── missing_heatmap.png
```

### Missingness per Feature

<img width="800" height="400" alt="missing_bar" src="https://github.com/user-attachments/assets/22bd4768-8b11-4141-8490-89ebd5cff9d1" />

This bar chart shows, for each column, the **proportion of missing entries**.

In the demo dataset, you should see:

* **`income`** and **`visits`** with the **highest bars** (~20% missing)
* **`age`** and **`score`** with shorter bars (~10% missing)
* **`target`** at **0% missing**

#### How to interpret this figure as a data scientist

* **High-missing features (`income`, `visits`)**

  * These may require more careful imputation (KNN or iterative)
  * If they are important predictors, poor imputation can heavily hurt model performance
  * In extreme real-world cases (>50% missing), you might even consider dropping the feature

* **Moderate-missing features (`age`, `score`)**

  * Simple imputation (median/mean) may be adequate
  * But you should check whether missingness is *random* or *systematic* (young users not reporting income)

* **0% missing label (`target`)**

  * This is ideal: you don’t want missing labels in supervised learning
  * If the label had missing values, you’d have to exclude those rows or treat it as a semi-supervised problem

This plot is your **first triage step**: it answers

> “Where is my dataset bleeding the most?”

---

### Missingness Matrix

<img width="1000" height="600" alt="missing_heatmap" src="https://github.com/user-attachments/assets/a902e326-91b6-4fd6-8ba6-613823b1f57e" />

This heatmap displays a **row × column matrix of missing values**:

* Each row = one sample (up to a capped number of rows for large datasets)
* Each column = one feature
* Colored cell = value is missing
* Blank cell = value is present

In the demo dataset, you should notice:

* For some rows, **only `income` is missing**
* For some rows, **only `visits` is missing**
* For one row, **`age` is missing** but other features are present
* For one row, **`score` is missing** while others are filled
* There is **no obvious block pattern** (like full rows of missing or a whole group of columns consistently missing together)

#### How to interpret this figure as a data scientist

This plot provides intuition about **missingness mechanisms**:

* **MCAR (Missing Completely At Random)**

  * Missingness appears scattered with no obvious pattern → the demo dataset roughly looks like this
  * In such cases, simple imputation strategies are usually less risky

* **MAR (Missing At Random)**

  * You might see patterns where missing values in one column align with values in another (low income → more missing `visits`)
  * This is a signal to investigate feature interactions before imputing

* **MNAR (Missing Not At Random)**

  * If missingness in a variable is *strongly related to its own (unobserved) values*
  * Harder to see visually; you’d need more careful statistical tests and domain knowledge

In practice, this matrix helps answer:

> “Do I have a random sprinkle of missing values, or is something structured (and dangerous) going on?”

---

## Imputation & Model Impact

Beyond visualization, Missing Data Doctor evaluates **how different imputations affect model performance**.

Currently, the CLI runs three strategies:

* `"simple"` → `SimpleImputer` (median / most frequent)
* `"knn"` → `KNNImputer` (nearest neighbors on numeric features)
* `"iterative"` → `IterativeImputer` (MICE-like multi-feature imputation)

Given a target and task (here: `target`, `classification`), the tool:

1. Imputes missing values using each strategy
2. Trains a `RandomForestClassifier` for each imputed dataset
3. Evaluates metrics on a held-out test set
4. Stores the results in:

```text
outputs/runs/demo/impact.json
```

Example structure (schema, not actual values):

```json
{
  "simple": {
    "AUC": 0.85,
    "Accuracy": 0.80
  },
  "knn": {
    "AUC": 0.87,
    "Accuracy": 0.82
  },
  "iterative": {
    "AUC": 0.86,
    "Accuracy": 0.81
  }
}
```

> The goal is not just “fill NA values”, but **quantify which imputation actually leads to a better model**.

---

## The HTML Report

The report template at:

```text
templates/report.html
```

is rendered with context including:

* `missing_summary`: list of columns with missing counts & percentages
* `missing_bar_path`: relative path to the bar chart, `plots/missing_bar.png`
* `missing_heatmap_path`: relative path to the heatmap, `plots/missing_heatmap.png`
* `impact`: metrics per imputation method (if a target is provided)

Inside the template, the figures are embedded like:

```html
<h3>Missingness per Feature</h3>
<img src="{{ missing_bar_path }}">

<h3>Missingness Matrix (Sampled Rows)</h3>
<img src="{{ missing_heatmap_path }}">
```

Because the report is saved inside the **same directory** as `plots/` (Option A), the relative paths:

```text
plots/missing_bar.png
plots/missing_heatmap.png
```

resolve correctly.

This makes every `outputs/runs/<run_name>/` folder **a self-contained artifact**:

* You can zip it
* Send it to someone
* They can open the HTML and see plots without editing anything

---

## CLI Usage Summary

Core CLI:

```cmd
python src\cli.py --data <path> --target <column> --task <classification|regression> --out_dir <run_folder>
```

Optional HTML report name (if you want a custom path instead of the default `missing_data_doctor.html`):

```cmd
python src\cli.py ^
  --data data\my_data.csv ^
  --target label ^
  --task classification ^
  --out_dir outputs\runs\experiment_01 ^
  --report outputs\runs\experiment_01\experiment_01_report.html
```

---

## Troubleshooting

### Figures not showing in HTML

* Make sure you **did not put the report into a different folder** than `out_dir`.
* With Option A (recommended), the report is inside `out_dir` and the images live in `out_dir/plots/`.
* Paths should be:

```html
<img src="plots/missing_bar.png">
<img src="plots/missing_heatmap.png">
```

### “ModuleNotFoundError: No module named 'pandas'”

Install dependencies:

```cmd
pip install pandas numpy scikit-learn matplotlib seaborn jinja2
```

### Virtual environment activation issues

If `.venv` is broken, delete it and recreate:

```cmd
rmdir /S /Q .venv
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Ideas for Extensions

You can extend Missing Data Doctor with:

* **More imputers**:

  * Median vs mean comparison
  * Domain-aware imputers (0 for missing count-based features)

* **Missingness, feature interaction analysis**:

  * Correlation between “is_missing(feature)” and numeric features
  * Logistic models predicting missingness as a function of other columns

* **Fairness & subgroup analysis**:

  * Does missingness disproportionately affect certain subgroups?

* **Time-aware gap analysis** (for time series):

  * Length and location of consecutive missing segments
