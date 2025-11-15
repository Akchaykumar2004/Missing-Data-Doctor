import argparse
import json
from pathlib import Path

import loaders
import profiling
import imputers
import impact as impact_mod
import viz
import report as report_mod

def main() -> None:
    ap = argparse.ArgumentParser(description="Missing Data Doctor (flat src version)")
    ap.add_argument("--data", required=True, help="Path to CSV with missing data")
    ap.add_argument("--target", default=None, help="Target column name (optional, required for model impact)")
    ap.add_argument("--task", choices=["classification", "regression"], default="classification")
    ap.add_argument("--out_dir", required=True, help="Output directory for metrics and plots")
    ap.add_argument("--report", default=None, help="Path to HTML report to generate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = loaders.load_csv(args.data)
    ms = profiling.missing_summary(df)
    ms.to_csv(out_dir / "missing_summary.csv", index=False)

    viz.missing_bar(df, str(plots_dir / "missing_bar.png"))
    viz.missing_heatmap(df, str(plots_dir / "missing_heatmap.png"))

    imputed_versions = {
        "simple": imputers.simple_impute(df.copy()),
        "knn": imputers.knn_impute(df.copy()),
        "iterative": imputers.iterative_impute(df.copy()),
    }

    impact_results = None
    if args.target:
        impact_results = impact_mod.evaluate_imputations(df.copy(), args.target, args.task, imputed_versions)
        with open(out_dir / "impact.json", "w", encoding="utf-8") as f:
            json.dump(impact_results, f, indent=2)

    summary = {"missing_summary": ms.to_dict(orient="records"), "impact": impact_results}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = Path(args.report) if args.report else (out_dir / "missing_data_doctor.html")
    context = {
        "missing_summary": ms.to_dict(orient="records"),
        "impact": impact_results,
        "missing_bar_path": f"{plots_dir.name}/missing_bar.png",
        "missing_heatmap_path": f"{plots_dir.name}/missing_heatmap.png",
        "task": args.task,
        "target": args.target,
    }
    report_mod.render_report(str(Path("templates") / "report.html"), str(report_path), context)
    print(f"Saved summary -> {out_dir / 'summary.json'}")
    print(f"Saved report  -> {report_path}")

if __name__ == "__main__":
    main()
