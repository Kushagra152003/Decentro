import os
import time
import glob
import yaml
import json
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))


def load_settings():
    with open(os.path.join(ROOT, "pipeline", "settings.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_collector(cfg):
    print("Running collector.")
    raw_dir = os.path.join(ROOT, cfg["paths"]["raw_dir"])
    os.makedirs(raw_dir, exist_ok=True)

    endpoints_pairs = [[e["url"], e["name"]] for e in cfg["endpoints"]]
    endpoints_json = json.dumps(endpoints_pairs)
    cmd = [
        "python",
        os.path.join(ROOT, "collector", "decentro_api_tester.py"),
        "--endpoints",
        endpoints_json,
        "--samples",
        str(cfg["collect"]["samples"]),
        "--sleep",
        str(cfg["collect"]["sleep_seconds"]),
        "--timeout",
        str(cfg["collect"].get("timeout_seconds", 10)),
        "--outdir",
        raw_dir,]
    subprocess.run(cmd, check=True)


def run_predictor(cfg):
    print("Running predictor.")
    results_dir = os.path.join(ROOT, cfg["paths"]["results_dir"])
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    raw_dir = os.path.join(ROOT, cfg["paths"]["raw_dir"])
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)


    targets = cfg.get("targets", None)
    targets_arg = []
    if targets:
        import json
        targets_arg = ["--targets_json", json.dumps(targets)]

    cmd = [
        "python",
        os.path.join(ROOT, "predictor", "decentro_ml_predictor.py"),
        "--raw_dir",
        raw_dir,
        "--results_dir",
        results_dir,
        "--processed_dir",
        processed_dir,
        *targets_arg,
    ]
    subprocess.run(cmd, check=True)


def print_latest_artifacts(cfg):
    results_dir = os.path.join(ROOT, cfg["paths"]["results_dir"])
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    pattern = os.path.join(results_dir, "decentro_ml_results_*.json")
    latest_json = None
    matches = sorted(glob.glob(pattern))
    if matches:
        latest_json = matches[-1]

    preds_csv = os.path.join(processed_dir, "predictions_export.csv")
    alerts_csv = os.path.join(processed_dir, "alerts_export.csv")

    print(f"Results JSON: {latest_json if latest_json else 'missing'}")
    print(f"Predictions CSV: {preds_csv} ({'exists' if os.path.exists(preds_csv) else 'missing'})")
    print(f"Alerts CSV: {alerts_csv} ({'exists' if os.path.exists(alerts_csv) else 'missing'})")


if __name__ == "__main__":
    cfg = load_settings()
    run_collector(cfg)
    run_predictor(cfg)
    print_latest_artifacts(cfg)
    print("Pipeline run complete. Launch the dashboard:")
    print(f"    streamlit run {os.path.join('app','app.py')}")