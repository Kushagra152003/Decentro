Project: Analysing Decentro Performance
This project measures how fast Decentro’s public pages feels to users, predicting when they might degrade, and turn them into actions. The pipeline checks the Homepage, Documentation, and API reference pages, compares them to competitions, and produces a dashboard that anyone can use to see what’s slow, what’s volatile, and what to fix first.

What it does:
Collects response times for key endpoints multiple times.
Analyzes averages, min/max, variability, trends, and anomalies for each endpoint.
Applies targets to judge weyher its good or needs work.
Generates alerts suggestions can be added.
Predicts short-term performance.
Outputs a Streamlit dashboard for visual aid.

Why it matters:
Homepage performance, first impression and click-through to docs.
Docs/API performance greatly increases developer experience, integration speed, and support load.
Consistent, fast endpoints reduce churn and strengthen competitive positioning.

How it works:
Collector pings each endpoint and writes a timestamped CSV.
Predictor loads the latest CSV, computes stats, produces alerts, and saves a results JSON + exports CSVs for predictions and alerts.
Dashboard reads the newest results and shows analysis, predictions, and action items. A button can run the pipeline on demand for fresh data after the pipeline is re-run.

What to look at in the dashboard:
Endpoint Performance Analysis: average time and variability per page.
ML Predictions: trend direction, strength, and the next 3 predicted timings.
Alert Feed: prioritized issues.

Typical use:
Run the pipeline → open the dashboard → see which endpoint is slow or unstable → apply quick fixes (CDN caching, defer third-party scripts, compress assets, pre-render docs) → run the pipeline again and confirm improvements.

P.S.:
A tiny “historical” seed exists so the dashboard isn’t empty on day zero(make sure to un-comment it before use); real measurements automatically override it once the collector runs.
Paths and endpoints are configurable without editing code.
The setup is intentionally lightweight and can later be scheduled or containerized.


Sample results can be seen here:-
https://go.screenpal.com/watch/cTQvqhnDBoo
https://go.screenpal.com/watch/cTQvqenDB2d
