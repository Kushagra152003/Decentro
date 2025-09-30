import argparse
import glob
import json
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

class DecentroPerformancePredictor:
    def __init__(self, raw_dir=None, results_dir=".", processed_dir=".", targets=None, anomaly_threshold=0.2):
        self.raw_dir = raw_dir
        self.results_dir = results_dir
        self.processed_dir = processed_dir
        self.anomaly_threshold = anomaly_threshold
        self.targets = targets or {
            "Homepage": 485,
            "Documentation": 424,
            "API_Overview": 424,
            "API_Basics": 424,}

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.df = None

    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_to_serializable(x) for x in obj]
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _load_latest_raw_csv(self):
        if not self.raw_dir:
            return None
        pattern = os.path.join(self.raw_dir, "decentro_api_performance_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        latest = files[-1]
        try:
            df_raw = pd.read_csv(latest)
            if not {"name", "time_ms", "timestamp"}.issubset(df_raw.columns):
                return None
            df_raw = df_raw.rename(columns={"name": "endpoint", "time_ms": "response_time"})
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            return df_raw[["timestamp", "endpoint", "response_time"]].copy()
        except Exception:
            return None

    def load_historical_data(self):
        df_raw = self._load_latest_raw_csv()
        if df_raw is not None and len(df_raw) > 0:
            self.df = df_raw.sort_values("timestamp").reset_index(drop=True)
            print(f"Loaded {len(self.df)} rows from latest raw CSV in {self.raw_dir}")
            return self.df
        
        historical_data = [
            {"timestamp": "2025-09-29T02:00:10", "endpoint": "Homepage", "response_time": 1433.45},
            {"timestamp": "2025-09-29T02:10:25", "endpoint": "Homepage", "response_time": 867.07},
            {"timestamp": "2025-09-29T02:11:24", "endpoint": "Homepage", "response_time": 675.00},
            {"timestamp": "2025-09-29T02:14:33", "endpoint": "Homepage", "response_time": 618.23},
            {"timestamp": "2025-09-29T02:31:07", "endpoint": "Homepage", "response_time": 897.92},

            {"timestamp": "2025-09-29T02:00:10", "endpoint": "Documentation", "response_time": 1379.04},
            {"timestamp": "2025-09-29T02:10:25", "endpoint": "Documentation", "response_time": 581.45},
            {"timestamp": "2025-09-29T02:11:24", "endpoint": "Documentation", "response_time": 488.99},
            {"timestamp": "2025-09-29T02:14:33", "endpoint": "Documentation", "response_time": 521.16},
            {"timestamp": "2025-09-29T02:31:07", "endpoint": "Documentation", "response_time": 1210.75},

            {"timestamp": "2025-09-29T02:14:33", "endpoint": "API_Overview", "response_time": 1410.92},
            {"timestamp": "2025-09-29T02:31:07", "endpoint": "API_Overview", "response_time": 1438.17},
            {"timestamp": "2025-09-29T02:14:33", "endpoint": "API_Basics", "response_time": 1452.64},]
        
        self.df = pd.DataFrame(historical_data)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        print(f"Loaded {len(self.df)} seeded performance data points")
        return self.df

    def analyze_patterns(self):
        analysis = {}
        for endpoint in self.df["endpoint"].unique():
            et = self.df[self.df["endpoint"] == endpoint]["response_time"]
            avg = float(et.mean())
            std = float(et.std()) if len(et) > 1 else 0.0
            analysis[endpoint] = {
                "avg_response_time": round(avg, 2),
                "min_response_time": round(float(et.min()), 2),
                "max_response_time": round(float(et.max()), 2),
                "variability": round(std, 2),
                "variability_pct": round((std / avg) * 100, 1) if avg > 0 else 0.0,
                "data_points": int(len(et)),}

        print("PERFORMANCE ANALYSIS:")
        for ep, st in analysis.items():
            print(f"\n {ep}:")
            print(f"   Average: {st['avg_response_time']}ms")
            print(f"   Range: {st['min_response_time']}ms - {st['max_response_time']}ms")
            print(f"   Variability: {st['variability_pct']}% (n={st['data_points']})")
            if st["avg_response_time"] > 1000:
                print(" ALERT: Average response time exceeds 1000ms")
            if st["variability_pct"] > 50:
                print(" WARNING: High performance variability")
        return analysis

    def predict_performance_issues(self):
        predictions = {}
        for endpoint in self.df["endpoint"].unique():
            df_ep = self.df[self.df["endpoint"] == endpoint].copy().reset_index(drop=True)
            n = len(df_ep)
            if n < 2:
                predictions[endpoint] = {
                    "trend_direction": "insufficient_data",
                    "predicted_next_3": [],
                    "anomalies_detected": 0,
                    "trend_strength": 0.0,}
                continue

            df_ep["time_order"] = np.arange(n, dtype=int)
            if n >= 3:
                X = df_ep[["time_order"]].values
                y = df_ep["response_time"].values
                model = LinearRegression().fit(X, y)
                future_X = np.array([[n + i] for i in range(1, 4)])
                future_preds = model.predict(future_X)
                anomalies = 0
                if n >= 4:
                    iso = IsolationForest(contamination=self.anomaly_threshold, random_state=42)
                    anomaly_labels = iso.fit_predict(df_ep[["response_time"]])
                    anomalies = int((anomaly_labels == -1).sum())

                predictions[endpoint] = {
                    "trend_direction": "increasing" if float(model.coef_[0]) > 0 else "decreasing",
                    "predicted_next_3": [round(float(p), 2) for p in future_preds],
                    "anomalies_detected": anomalies,
                    "trend_strength": round(float(abs(model.coef_[0])), 2),}
            else:
                avg = float(df_ep["response_time"].mean())
                predictions[endpoint] = {
                    "trend_direction": "stable",
                    "predicted_next_3": [round(avg, 2)] * 3,
                    "anomalies_detected": 0,
                    "trend_strength": 0.0,}
        return predictions

    def generate_business_alerts(self, predictions, analysis):
        alerts = []
        for ep, pred in predictions.items():
            if ep not in analysis:
                continue
            avg = analysis[ep]["avg_response_time"]
            var = analysis[ep]["variability_pct"]

            if avg > 1500:
                alerts.append({
                    "level": "CRITICAL", "endpoint": ep,
                    "message": f"{ep} averaging {avg}ms - User churn risk",
                    "action": "IMMEDIATE action required",
                    "priority": 1,})
            elif avg > 1000:
                alerts.append({
                    "level": "WARNING", "endpoint": ep,
                    "message": f"{ep} averaging {avg}ms - Above best practice (<500ms)",
                    "action": "Schedule action within 2 weeks",
                    "priority": 2,})

            if var > 60:
                alerts.append({
                    "level": "STABILITY", "endpoint": ep,
                    "message": f"{ep} variability {var}% - inconsistent experience",
                    "action": "--action can be delayed--",
                    "priority": 2,})

            if ep in self.targets and avg > self.targets[ep]:
                gap = round(((avg / self.targets[ep]) - 1) * 100, 0)
                competitor = "Razorpay" if ep in ["Documentation", "API_Overview", "API_Basics"] else "market avg"
                alerts.append({
                    "level": "COMPETITIVE", "endpoint": ep,
                    "message": f"{ep} {gap}% slower than {competitor} ({self.targets[ep]}ms)",
                    "action": f"Optimize to ≤ {self.targets[ep]}ms for parity",
                    "priority": 1 if gap > 100 else 2,})

            if pred["trend_direction"] == "increasing" and pred["trend_strength"] > 10:
                alerts.append({
                    "level": "TREND_ALERT", "endpoint": ep,
                    "message": f"{ep} degrading (+{pred['trend_strength']}ms per measurement)",
                    "action": "Monitor actions",
                    "priority": 2,})

        alerts.sort(key=lambda x: x["priority"])
        return alerts

    def calculate_business_impact(self, analysis):
        doc = analysis.get("Documentation", {}).get("avg_response_time", 1000)
        api = analysis.get("API_Overview", {}).get("avg_response_time", 1000)
        doc_score = max(0, 100 - (doc - 300) / 10)
        api_score = max(0, 100 - (api - 300) / 10)
        impact = {
            "User_experience_score": round((doc_score + api_score) / 2, 1),
            "estimated_churn_risk": "HIGH" if (doc > 1000 or api > 1000) else "MEDIUM" if (doc > 600 or api > 600) else "LOW",
            "competitive_position": "Leading" if doc < 500 else "Competitive" if doc < 800 else "Lagging",
        }
        return impact

    def run_complete_analysis(self):
        print("DECENTRO PERFORMANCE PREDICTION & BUSINESS INTELLIGENCE")
        print("/n")
        self.load_historical_data()
        analysis = self.analyze_patterns()
        predictions = self.predict_performance_issues()

        print("\n ML PERFORMANCE PREDICTIONS:")
        print("-" * 35)
        for ep, pred in predictions.items():
            print(f"\n{ep}:")
            print(f"   Trend: {pred['trend_direction']}")
            if pred["predicted_next_3"]:
                print(f"   Next 3: {[f'{p}ms' for p in pred['predicted_next_3']]}")
            print(f"   Anomalies: {pred['anomalies_detected']}")
            if pred["trend_strength"] > 0:
                print(f"   Trend Strength: {pred['trend_strength']}ms/measurement")

        alerts = self.generate_business_alerts(predictions, analysis)
        impact = self.calculate_business_impact(analysis)

        print("\n BUSINESS ALERTS & RECOMMENDATIONS:")
        print("-" * 40)
        if alerts:
            for a in alerts:
                wrn = {
                    "CRITICAL": "CRITICAL",
                    "WARNING": "WARNING",
                    "COMPETITIVE": "COMPETITIVE",
                    "STABILITY": "STABILITY",
                    "TREND_ALERT": "TREND_ALERT",
                }.get(a["level"],)
                print(f"\n{wrn} {a['level']} - Priority {a['priority']}")
                print(f"   Endpoint: {a['endpoint']}")
                print(f"   Issue: {a['message']}")
                print(f"   Action: {a['action']}")
        else:
            print("No critical issues detected")

        print("\n BUSINESS IMPACT:")
        print("-" * 18)
        print(f"User Experience: {impact['User_experience_score']}/100")
        print(f"Churn Risk: {impact['estimated_churn_risk']}")
        print(f"Competitive Position: {impact['competitive_position']}")

        return {
            "analysis": analysis,
            "predictions": predictions,
            "alerts": alerts,
            "business_impact": impact,}

    def export_csvs(self, results):
        preds = []
        for ep, pred in results["predictions"].items():
            preds.append(
                {
                    "endpoint": ep,
                    "trend": pred["trend_direction"],
                    "next_1_ms": (pred["predicted_next_3"] + [None])[0] if pred["predicted_next_3"] else None,
                    "next_2_ms": (pred["predicted_next_3"] + [None, None])[1] if pred["predicted_next_3"] else None,
                    "next_3_ms": (pred["predicted_next_3"] + [None, None, None])[2] if pred["predicted_next_3"] else None,
                    "anomalies": pred["anomalies_detected"],
                    "trend_strength": pred["trend_strength"],
                })
        pd.DataFrame(preds).to_csv(os.path.join(self.processed_dir, "predictions_export.csv"), index=False)

        pd.DataFrame(results["alerts"]).to_csv(os.path.join(self.processed_dir, "alerts_export.csv"), index=False)

    def export_json(self, results):
        serializable = self.convert_to_serializable(
            {
                "timestamp": datetime.now().isoformat(),
                "analysis": results["analysis"],
                "predictions": results["predictions"],
                "alerts": results["alerts"],
                "business_impact": results["business_impact"],
            })
        filename = f"decentro_ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path = os.path.join(self.results_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n Results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Decentro ML Predictor (pipeline-friendly)")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory with raw CSVs (data/raw)")
    parser.add_argument("--results_dir", type=str, default=".", help="Where to write JSON results (data/results)")
    parser.add_argument("--processed_dir", type=str, default=".", help="Where to write predictions/alerts CSVs (data/processed)")
    parser.add_argument("--targets_json", type=str, default="", help='Optional JSON string or file with targets, e.g. {"Homepage":485,...}')
    return parser.parse_args()


def load_targets(targets_json):
    if not targets_json:
        return None
    if os.path.exists(targets_json):
        with open(targets_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(targets_json)



if __name__ == "__main__":
    args = parse_args()
    targets = load_targets(args.targets_json)

    predictor = DecentroPerformancePredictor(
        raw_dir=args.raw_dir,
        results_dir=args.results_dir,
        processed_dir=args.processed_dir,
        targets=targets,)
    
    results = predictor.run_complete_analysis()
    predictor.export_csvs(results)
    predictor.export_json(results)

    def slo_compliance(df, endpoint, target_ms=500):
        et = df[df["endpoint"] == endpoint]["response_time"]
        if len(et) == 0:
            return 0.0
        return round((et.lt(target_ms).mean()) * 100, 1)

    for ep in results["analysis"].keys():
        target = predictor.targets.get(ep, 500)
        print(f"SLO compliance ({ep} @ {target}ms): {slo_compliance(predictor.df, ep, target)}%")

    print("ML predictor run complete!")