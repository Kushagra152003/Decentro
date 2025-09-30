import streamlit as st
import pandas as pd
import json
import glob
from datetime import datetime
import plotly.express as px
import os
#Abs path
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(app_dir, os.pardir))
os.chdir(project_root)


st.set_page_config(page_title="Decentro Performance Dashboard", layout="wide")

st.title("Decentro Performance Dashboard")
st.caption("Live benchmarks to show current performance")

#Loading data
@st.cache_data(show_spinner=False)
def load_json_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)
st.sidebar.header("Data Sources")

json_pattern = st.sidebar.text_input(
    "Results JSON pattern",
    value="data/results/decentro_ml_results_*.json")
pred_csv = st.sidebar.text_input(
    "Predictions CSV",
    value="data/processed/predictions_export.csv")
alerts_csv = st.sidebar.text_input(
    "Alerts CSV",
    value="data/processed/alerts_export.csv")
raw_pattern = st.sidebar.text_input(
    "Raw Response CSV pattern",
    value="data/raw/decentro_api_performance_*.csv")


json_files = sorted(glob.glob(json_pattern))
selected_json = json_files[-1] if json_files else None
raw_files = sorted(glob.glob(raw_pattern))
selected_raw = raw_files[-1] if raw_files else None

results = None
if selected_json:
    try:
        results = load_json_results(selected_json)
        st.sidebar.success(f"Loaded: {selected_json}")
    except Exception as e:
        st.sidebar.error(f"Failed to load results JSON: {e}")
else:
    st.sidebar.warning("No results JSON found.")

try:
    pred_df = load_csv(pred_csv)
except Exception:
    pred_df = pd.DataFrame()

try:
    alerts_df = load_csv(alerts_csv)
except Exception:
    alerts_df = pd.DataFrame()

try:
    if selected_raw:
        raw_df = load_csv(selected_raw)
    else:
        raw_df = pd.DataFrame()
except Exception:
    raw_df = pd.DataFrame()


col1, col2, col3 = st.columns(3)
if results and 'business_impact' in results:
    bi = results['business_impact']
    col1.metric("User Experience", f"{bi['User_experience_score']}/100")
    col2.metric("Churn Risk", bi['estimated_churn_risk'])
    col3.metric("Competitive Position", bi['competitive_position'])
else:
    col1.metric("User Experience", "-")
    col2.metric("Churn Risk", "-")
    col3.metric("Competitive Position", "-")

st.divider()

st.subheader("Endpoint Performance Analysis")
if results and 'analysis' in results:
    analysis_df = pd.DataFrame(results['analysis']).T.reset_index().rename(columns={'index': 'endpoint'})
    st.dataframe(analysis_df, use_container_width=True)
else:
    st.info("Run ML predictor.")

st.subheader("ML Predictions")
if results and 'predictions' in results:
    pred_list = []
    for ep, p in results['predictions'].items():
        pred_list.append({
            'endpoint': ep,
            'trend': p.get('trend_direction', '-'),
            'trend_strength': p.get('trend_strength', 0),
            'next_1ms': (p.get('predicted_next_3') or [None])[0],
            'next_2ms': (p.get('predicted_next_3') or [None, None])[1] if len(p.get('predicted_next_3') or []) > 1 else None,
            'next_3ms': (p.get('predicted_next_3') or [None, None, None])[2] if len(p.get('predicted_next_3') or []) > 2 else None,
            'anomalies': p.get('anomalies_detected', 0)
        })
    pred_table = pd.DataFrame(pred_list)
    st.dataframe(pred_table, use_container_width=True)
else:
    st.info("Predictions will appear after ML run.")

st.subheader("Alert Feed")
if not alerts_df.empty:
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.info("Export alerts_export.csv.")

st.divider()

st.subheader("Performance vs target graph")
if results and 'analysis' in results:
    analysis_df = pd.DataFrame(results['analysis']).T.reset_index().rename(columns={'index': 'endpoint'})
    analysis_df['target_ms'] = analysis_df['endpoint'].apply(lambda x: 424 if 'API' in x or 'Documentation' in x else 485)
    chart_df = analysis_df[['endpoint', 'avg_response_time', 'target_ms']]
    fig = px.bar(
        chart_df.melt(id_vars='endpoint', var_name='metric', value_name='ms'),
        x='endpoint', y='ms', color='metric', barmode='group',
        title='Average Response Time vs Target'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run ML to visualize performance.")

st.divider()
if st.sidebar.button("Refresh latest"):
    st.rerun()

if selected_json:
    st.caption(f"Using results from: {selected_json}")