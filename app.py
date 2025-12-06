# ===============================================================
# BITCOIN PRICE PREDICTION – FLASK WEB APPLICATION
# Beautiful UI + Interactive Charts + Stable Prediction Engine
# ===============================================================

# -------------------------------------------------------
# IMPORTANT: LIMIT TENSORFLOW RESOURCE USAGE (MUST BE FIRST)
# -------------------------------------------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import plotly.graph_objs as go
import plotly.offline as plot
import pandas as pd

# -------------------------------------------------------
# INITIALIZE FLASK APP
# -------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------
# FILE PATHS (ABSOLUTE FOR SAFETY)
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
LAST60_PATH = os.path.join(BASE_DIR, "model", "last_60_scaled.npy")
HISTORY_PATH = os.path.join(BASE_DIR, "model", "btc_history.csv")
TRAINING_HISTORY = os.path.join(BASE_DIR, "model", "training_history.csv")

# -------------------------------------------------------
# GLOBAL VARIABLES (LAZY LOADED)
# -------------------------------------------------------
model = None
scaler = None
last_60 = None
time_step = 60

# -------------------------------------------------------
# LAZY LOADER FUNCTION (CRITICAL FIX)
# -------------------------------------------------------
def load_resources():
    global model, scaler, last_60
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
    if scaler is None:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    if last_60 is None:
        last_60 = np.load(LAST60_PATH).reshape(60,)
    return model, scaler, last_60

# ========================================================================
# HOME PAGE – Dashboard Overview
# ========================================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        message="✅ Application Loaded Successfully! Choose an action from the menu."
    )

# ========================================================================
# PREDICT NEXT "X" DAYS OF BITCOIN PRICE — WITH REAL FUTURE DATES
# ========================================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        days = int(request.form["days"])
    except:
        return render_template("index.html", message="❌ Invalid input.")

    if days < 1 or days > 365:
        return render_template("index.html", message="❌ Enter days between 1–365.")

    # ✅ Load model & scaler only when needed
    model, scaler, last_60_local = load_resources()

    # ---------------------------------------------------------
    # LOAD LAST DATE FROM HISTORICAL CSV
    # ---------------------------------------------------------
    df_hist = pd.read_csv(HISTORY_PATH)
    df_hist.columns = df_hist.columns.str.replace(r"\s+", "", regex=True)
    df_hist["Date"] = pd.to_datetime(df_hist["Date"])

    last_date = df_hist["Date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days
    )

    # ---------------------------------------------------------
    # PREDICTION LOOP
    # ---------------------------------------------------------
    temp_input = list(last_60_local)
    lst_output = []

    for _ in range(days):
        x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)

        temp_input.append(float(yhat[0][0]))
        lst_output.append(float(yhat[0][0]))

    final_pred = scaler.inverse_transform(
        np.array(lst_output).reshape(-1, 1)
    ).flatten()

    # ---------------------------------------------------------
    # INTERACTIVE DATE-BASED GRAPH
    # ---------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=final_pred,
        mode="lines+markers",
        line=dict(color="#ff9900", width=3),
        marker=dict(size=6),
        name="Predicted Price"
    ))

    fig.update_layout(
        title=f"Bitcoin Price Prediction ({future_dates[0].date()} → {future_dates[-1].date()})",
        xaxis_title="Date",
        yaxis_title="Predicted Price (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(size=15)
    )

    graph_html = plot.plot(fig, output_type="div")
    latest_price = f"${final_pred[-1]:,.2f}"

    return render_template(
        "index.html",
        prediction=latest_price,
        graph=graph_html,
        days=days,
        message="✅ Prediction Completed Successfully!"
    )

# ========================================================================
# SHOW BITCOIN PRICE HISTORY (INTERACTIVE)
# ========================================================================
@app.route("/history")
def history():
    df = pd.read_csv(HISTORY_PATH)
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)
    df["Date"] = pd.to_datetime(df["Date"])

    close_col = [c for c in df.columns if "close" in c.lower()][0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df[close_col],
        mode="lines",
        line=dict(color="#1f77b4", width=2),
        name="Close Price"
    ))

    fig.update_layout(
        title="Bitcoin Historical Close Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    graph_html = plot.plot(fig, output_type="div")

    return render_template(
        "index.html",
        history_graph=graph_html,
        message="✅ Historical Price Data Loaded!"
    )

# ========================================================================
# TRAINING VS VALIDATION CURVE
# ========================================================================
@app.route("/training")
def training_curve():
    df = pd.read_csv(TRAINING_HISTORY)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(y=df["val_loss"], mode="lines", name="Validation Loss"))

    fig.update_layout(
        title="Model Training vs Validation Loss",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    graph_html = plot.plot(fig, output_type="div")

    return render_template(
        "index.html",
        training_graph=graph_html,
        message="✅ Training Curve Loaded!"
    )

# ========================================================================
# ANALYSIS DASHBOARD — ALL VISUALS
# ========================================================================
@app.route("/analysis")
def analysis():
    df = pd.read_csv(HISTORY_PATH)
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    # 1️⃣ FULL PRICE CHART
    fig1 = go.Figure()
    for col in ["Open", "High", "Low", "Close"]:
        fig1.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))

    fig1.update_layout(
        title="📊 Bitcoin Price Analysis (2018–2025)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    full_chart = plot.plot(fig1, output_type="div")

    # 2️⃣ MONTHLY HIGH / LOW
    monthly = df.groupby("Month").agg({"High": "max", "Low": "min"}).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly["Month"], y=monthly["High"], name="Monthly High"))
    fig2.add_trace(go.Bar(x=monthly["Month"], y=monthly["Low"], name="Monthly Low"))

    fig2.update_layout(
        title="📈 Month-wise High vs Low Price",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    month_chart = plot.plot(fig2, output_type="div")

    # 3️⃣ MONTHLY OPEN / CLOSE
    monthly_oc = df.groupby("Month").agg({"Open": "mean", "Close": "mean"}).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=monthly_oc["Month"], y=monthly_oc["Open"], name="Open Price"))
    fig3.add_trace(go.Bar(x=monthly_oc["Month"], y=monthly_oc["Close"], name="Close Price"))

    fig3.update_layout(
        title="📉 Monthly Open vs Close Comparison",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    open_close_chart = plot.plot(fig3, output_type="div")

    return render_template(
        "index.html",
        analysis_full=full_chart,
        analysis_high_low=month_chart,
        analysis_open_close=open_close_chart,
        message="✅ Analysis Dashboard Loaded!"
    )

# ========================================================================
# RUN APP (LOCAL ONLY)
# ========================================================================
if __name__ == "__main__":
    app.run(debug=True)