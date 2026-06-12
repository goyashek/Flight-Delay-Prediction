"""
Flight Delay Prediction — Streamlit App
----------------------------------------
Loads the trained pipeline from flight_delay_model.pkl
and provides a clean UI for single-flight prediction.

Run with:
    streamlit run app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD ARTIFACT
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str = "flight_delay_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    artifact = load_model()
except FileNotFoundError:
    st.error(
        "**Model file not found.**  \n"
        "Make sure `flight_delay_model.pkl` is in the same folder as `app.py`.  \n"
        "Run Section 13 of the portfolio notebook to generate it."
    )
    st.stop()

# Unpack frequently used items
pipeline              = artifact["pipeline"]
airline_codes         = artifact["airline_codes"]
origin_codes          = artifact["origin_codes"]
dest_codes            = artifact["dest_codes"]
dep_time_blocks       = artifact["dep_time_blocks"]
airport_lookup        = artifact["airport_lookup"]
airline_lookup        = artifact["airline_lookup"]
airline_hist_map      = artifact["airline_hist_map"]
origin_hist_map       = artifact["origin_hist_map"]
avg_taxiout_map       = artifact["avg_taxiout_map"]
avg_daily_flights_map = artifact["avg_daily_flights_map"]
global_delay_rate     = artifact["global_delay_rate"]
global_taxiout        = artifact["global_taxiout"]
global_daily_flights  = artifact["global_daily_flights"]
model_name            = artifact["model_name"]
metrics               = artifact["metrics"]
NUMERIC_FEATURES      = artifact["numeric_features"]
CATEGORICAL_FEATURES  = artifact["categorical_features"]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April",
    5:"May",     6:"June",     7:"July",  8:"August",
    9:"September", 10:"October", 11:"November", 12:"December",
}

DAY_NAMES = {
    0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday",
    4:"Friday", 5:"Saturday", 6:"Sunday",
}

AIRLINE_FULL = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "NK": "Spirit Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
}

def fmt_airport(code: str) -> str:
    city = airport_lookup.get(code, "")
    return f"{code} — {city}" if city else code

def fmt_airline(code: str) -> str:
    name = AIRLINE_FULL.get(code, airline_lookup.get(code, ""))
    return f"{code} — {name}" if name else code

def hhmm_to_int(hh: int, mm: int) -> int:
    """Convert hour+minute to HHMM integer (e.g. 14:30 → 1430)."""
    return hh * 100 + mm

def get_season(month: int) -> str:
    return {12:"Winter", 1:"Winter", 2:"Winter",
             3:"Spring", 4:"Spring", 5:"Spring",
             6:"Summer", 7:"Summer", 8:"Summer",
             9:"Fall",  10:"Fall",  11:"Fall"}[month]

def is_holiday_window(month: int, day: int) -> int:
    return int(
        (month == 12 and day >= 15) or
        (month ==  1 and day <=  7) or
        (month ==  6 and day >= 15) or
        (month ==  7 and day <=  7) or
        (month == 11 and day >= 20)
    )

def build_feature_row(
    airline, origin, dest,
    dep_hh, dep_mm,
    arr_hh, arr_mm,
    elapsed_min, distance,
    dep_time_blk,
    month, day_of_month, day_of_week,
) -> pd.DataFrame:
    """Reproduce all feature engineering done during training."""

    dep_time_int = hhmm_to_int(dep_hh, dep_mm)
    arr_time_int = hhmm_to_int(arr_hh, arr_mm)
    dep_hour     = dep_hh

    is_weekend   = int(day_of_week in (5, 6))
    is_peak_hour = int(dep_hour in list(range(7, 10)) + list(range(16, 20)))
    season       = get_season(month)
    holiday      = is_holiday_window(month, day_of_month)

    # Lookup table features — fall back to globals for unseen airports/airlines
    airline_hist = airline_hist_map.get(airline, global_delay_rate)
    origin_hist  = origin_hist_map.get(origin,  global_delay_rate)
    avg_taxiout  = avg_taxiout_map.get(origin,  global_taxiout)
    daily_flights = avg_daily_flights_map.get(origin, global_daily_flights)

    # Cyclic encodings
    dep_hour_sin    = np.sin(2 * np.pi * dep_hour    / 24)
    dep_hour_cos    = np.cos(2 * np.pi * dep_hour    / 24)
    month_sin       = np.sin(2 * np.pi * month       / 12)
    month_cos       = np.cos(2 * np.pi * month       / 12)
    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

    distance_bucket_bins   = [0, 300, 600, 1000, 1500, 9999]
    distance_bucket_labels = ["Short", "Medium", "Long", "X-Long", "Ultra"]
    dist_bucket = pd.cut(
        [distance],
        bins=distance_bucket_bins,
        labels=distance_bucket_labels
    )[0]

    row = {
        # Numeric
        "CRSDepTime":          dep_time_int,
        "CRSArrTime":          arr_time_int,
        "CRSElapsedTime":      elapsed_min,
        "Distance":            distance,
        "Month":               month,
        "DayofMonth":          day_of_month,
        "DayOfWeek":           day_of_week,
        "DepHour":             dep_hour,
        "IsWeekend":           is_weekend,
        "IsPeakHour":          is_peak_hour,
        "HolidayWindow":       holiday,
        "OriginDailyFlights":  daily_flights,
        "AirlineHistDelayRate": airline_hist,
        "OriginHistDelayRate": origin_hist,
        "AvgOriginTaxiOut":    avg_taxiout,
        "DepHour_sin":         dep_hour_sin,
        "DepHour_cos":         dep_hour_cos,
        "Month_sin":           month_sin,
        "Month_cos":           month_cos,
        "DayOfWeek_sin":       day_of_week_sin,
        "DayOfWeek_cos":       day_of_week_cos,
        # Categorical
        "Marketing_Airline_Network": airline,
        "Origin":              origin,
        "Dest":                dest,
        "DistanceBucket":      str(dist_bucket),
        "Season":              season,
        "DepTimeBlk":          dep_time_blk,
    }

    # Only keep columns the pipeline was trained on
    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame([{k: row.get(k, np.nan) for k in all_cols}])


def risk_colour(proba: float):
    if proba < 0.30:
        return "#28a745", "LOW RISK"
    elif proba < 0.55:
        return "#ffc107", "MODERATE RISK"
    else:
        return "#dc3545", "HIGH RISK"


# ─────────────────────────────────────────────
# SIDEBAR — MODEL INFO
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/100px-PNG_transparency_demonstration_1.png",
             width=60, caption="")  # placeholder — replace with your logo
    st.title("About This Model")
    st.markdown(f"**Algorithm:** {model_name}")
    st.markdown(f"**Training rows:** {artifact['train_rows']:,}")
    st.markdown(f"**Test rows:** {artifact['test_rows']:,}")

    st.markdown("---")
    st.markdown("### Test-Set Performance")
    if model_name in metrics:
        m = metrics[model_name]
        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
        col2.metric("F1 Score", f"{m['F1']:.3f}")
        col1.metric("Recall",   f"{m['Recall']:.3f}")
        col2.metric("Precision", f"{m['Precision']:.3f}")

    st.markdown("---")
    st.markdown("### All Models Compared")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "AUC":   round(m["ROC-AUC"], 3),
            "F1":    round(m["F1"],      3),
            "Recall": round(m["Recall"], 3),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.caption(
        "Target: ArrDel15 = 1 when arrival delay > 15 min.  "
        "Features use only scheduling-time info — no post-departure leakage."
    )

# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
st.title("✈️ Flight Delay Predictor")
st.markdown(
    "Enter flight scheduling details below. The model predicts whether your flight "
    "will arrive **more than 15 minutes late**."
)
st.markdown("---")

# ── INPUT FORM ────────────────────────────────
with st.form("prediction_form"):
    st.subheader("Flight Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        airline_label = st.selectbox(
            "Airline",
            options=airline_codes,
            format_func=fmt_airline,
            index=airline_codes.index("DL") if "DL" in airline_codes else 0,
        )

    with col2:
        origin_label = st.selectbox(
            "Origin Airport",
            options=origin_codes,
            format_func=fmt_airport,
            index=origin_codes.index("ATL") if "ATL" in origin_codes else 0,
        )

    with col3:
        dest_label = st.selectbox(
            "Destination Airport",
            options=dest_codes,
            format_func=fmt_airport,
            index=dest_codes.index("ORD") if "ORD" in dest_codes else 0,
        )

    st.markdown("#### Schedule")
    col4, col5, col6 = st.columns(3)

    with col4:
        dep_hh = st.slider("Departure Hour (24h)", 0, 23, 9)
        dep_mm = st.selectbox("Departure Minute", [0, 15, 30, 45], index=0)

    with col5:
        arr_hh = st.slider("Arrival Hour (24h)", 0, 23, 11)
        arr_mm = st.selectbox("Arrival Minute", [0, 15, 30, 45], index=0)

    with col6:
        elapsed_min = st.number_input(
            "Scheduled Flight Duration (min)", min_value=20, max_value=700, value=120, step=5
        )

    st.markdown("#### Route Info")
    col7, col8 = st.columns(2)

    with col7:
        distance = st.number_input(
            "Route Distance (miles)", min_value=30, max_value=5000, value=800, step=10
        )
        dep_time_blk_sel = st.selectbox(
            "Departure Time Block",
            options=dep_time_blocks if dep_time_blocks else ["0900-0959"],
            index=2 if len(dep_time_blocks) > 2 else 0,
        )

    with col8:
        month = st.selectbox(
            "Month",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda m: MONTH_NAMES[m],
            index=0,
        )
        day_of_month = st.slider("Day of Month", 1, 31, 15)
        day_of_week  = st.selectbox(
            "Day of Week",
            options=list(DAY_NAMES.keys()),
            format_func=lambda d: DAY_NAMES[d],
            index=0,
        )

    submitted = st.form_submit_button("🔍 Predict Delay", use_container_width=True)

# ── PREDICTION OUTPUT ─────────────────────────
if submitted:
    X_input = build_feature_row(
        airline=airline_label,
        origin=origin_label,
        dest=dest_label,
        dep_hh=dep_hh, dep_mm=dep_mm,
        arr_hh=arr_hh, arr_mm=arr_mm,
        elapsed_min=elapsed_min,
        distance=distance,
        dep_time_blk=dep_time_blk_sel,
        month=month,
        day_of_month=day_of_month,
        day_of_week=day_of_week,
    )

    prediction = pipeline.predict(X_input)[0]
    proba      = pipeline.predict_proba(X_input)[0, 1]
    colour, risk_label = risk_colour(proba)

    st.markdown("---")
    st.subheader("Prediction Result")

    res_col1, res_col2, res_col3 = st.columns([2, 2, 3])

    with res_col1:
        if prediction == 1:
            st.error("### 🔴 DELAYED\nFlight likely to arrive **>15 min late**")
        else:
            st.success("### 🟢 ON TIME\nFlight likely to arrive **within 15 min**")

    with res_col2:
        st.markdown(f"""
        <div style='border-left: 6px solid {colour}; padding: 12px 20px;
                    border-radius: 8px; background: #1e1e2e;'>
            <div style='font-size:2.2rem; font-weight:800; color:{colour};'>{proba*100:.1f}%</div>
            <div style='color:#ccc; font-size:0.95rem;'>Delay Probability</div>
            <div style='font-size:1.1rem; font-weight:700; color:{colour}; margin-top:6px;'>{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        # Progress-bar style gauge
        bar_pct = int(proba * 100)
        st.markdown("**Delay Probability Gauge**")
        st.progress(proba)
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.8rem;'>"
            f"<span style='color:#28a745'>0% — Very Safe</span>"
            f"<span style='color:#dc3545'>100% — Very Likely</span>"
            f"</div>", unsafe_allow_html=True
        )

    # ── Feature summary card ──────────────────
    st.markdown("---")
    st.subheader("Flight Profile Summary")

    season     = get_season(month)
    is_holiday = bool(is_holiday_window(month, day_of_month))
    is_peak    = dep_hh in list(range(7, 10)) + list(range(16, 20))
    is_weekend = day_of_week in (5, 6)

    tags = []
    if is_holiday: tags.append("🎄 Holiday Window")
    if is_peak:    tags.append("⚡ Peak Hour")
    if is_weekend: tags.append("📅 Weekend")
    tags.append(f"🌤 {season}")

    tag_html = " &nbsp; ".join(
        f"<span style='background:#333; padding:4px 10px; border-radius:12px; font-size:0.85rem;'>{t}</span>"
        for t in tags
    )
    st.markdown(tag_html, unsafe_allow_html=True)
    st.markdown("")

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    info_col1.metric("Route",    f"{origin_label} → {dest_label}")
    info_col2.metric("Airline",  fmt_airline(airline_label))
    info_col3.metric("Distance", f"{distance:,} mi")
    info_col4.metric("Duration", f"{elapsed_min} min")

    # ── Risk factors breakdown ────────────────
    st.markdown("---")
    st.subheader("Risk Factor Analysis")

    airline_rate = airline_hist_map.get(airline_label, global_delay_rate)
    origin_rate  = origin_hist_map.get(origin_label,   global_delay_rate)
    avg_to       = avg_taxiout_map.get(origin_label,   global_taxiout)

    factor_data = {
        "Risk Factor":    ["Airline Delay History", "Airport Delay History",
                           "Departure Hour Risk",   "Holiday / Season Effect",
                           "Avg Taxi-Out Time"],
        "Value":          [f"{airline_rate*100:.1f}%", f"{origin_rate*100:.1f}%",
                           f"Hour {dep_hh:02d}:00 ({'Peak' if is_peak else 'Off-peak'})",
                           f"{'YES' if is_holiday else 'No'} / {season}",
                           f"{avg_to:.1f} min"],
        "Impact":         [
            "🔴 High" if airline_rate > 0.30 else "🟡 Medium" if airline_rate > 0.20 else "🟢 Low",
            "🔴 High" if origin_rate  > 0.30 else "🟡 Medium" if origin_rate  > 0.20 else "🟢 Low",
            "🔴 High" if is_peak               else "🟢 Low",
            "🔴 High" if is_holiday             else "🟡 Medium" if season in ("Summer","Winter") else "🟢 Low",
            "🔴 High" if avg_to > 20            else "🟡 Medium" if avg_to > 15 else "🟢 Low",
        ]
    }
    st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

    # ── Advice box ───────────────────────────
    st.markdown("---")
    if proba >= 0.55:
        st.warning(
            "**High delay risk detected.**  \n"
            "Consider booking an earlier departure, choosing a more reliable airline, "
            "or allowing extra connection time. Evening flights at congested hubs are "
            "most susceptible to cascade delays."
        )
    elif proba >= 0.30:
        st.info(
            "**Moderate delay risk.**  \n"
            "This flight has some risk factors but is not in the high-risk zone. "
            "Monitor the flight status on the day of travel."
        )
    else:
        st.success(
            "**Low delay risk.**  \n"
            "This flight profile looks solid. Early-morning departures from reliable "
            "airlines at low-congestion airports have the best on-time performance."
        )

# ── BATCH PREDICTION ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Batch Prediction (CSV Upload)")
st.markdown(
    "Upload a CSV with the same columns as the training data. "
    "The app will score every row and let you download the results."
)

uploaded = st.file_uploader("Upload flight CSV", type=["csv"])
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df):,} rows × {batch_df.shape[1]} columns.")

        # ── Reproduce feature engineering on the batch ─────────────────
        batch_df['FlightDate'] = pd.to_datetime(batch_df['FlightDate'], errors='coerce')
        batch_df['Year']       = batch_df['FlightDate'].dt.year
        batch_df['Month']      = batch_df['FlightDate'].dt.month
        batch_df['DayofMonth'] = batch_df['FlightDate'].dt.day
        batch_df['DayOfWeek']  = batch_df['FlightDate'].dt.dayofweek

        batch_df['DepHour']    = (batch_df['CRSDepTime'] // 100).clip(0, 23)
        batch_df['IsWeekend']  = batch_df['DayOfWeek'].isin([5, 6]).astype(int)
        batch_df['IsPeakHour'] = batch_df['DepHour'].isin(
            list(range(7, 10)) + list(range(16, 20))
        ).astype(int)
        batch_df['Season']     = batch_df['Month'].map(
            {12:'Winter',1:'Winter',2:'Winter',
              3:'Spring',4:'Spring',5:'Spring',
              6:'Summer',7:'Summer',8:'Summer',
              9:'Fall',10:'Fall',11:'Fall'}
        )
        batch_df['HolidayWindow'] = (
            ((batch_df['Month'] == 12) & (batch_df['DayofMonth'] >= 15)) |
            ((batch_df['Month'] ==  1) & (batch_df['DayofMonth'] <=  7)) |
            ((batch_df['Month'] ==  6) & (batch_df['DayofMonth'] >= 15)) |
            ((batch_df['Month'] ==  7) & (batch_df['DayofMonth'] <=  7)) |
            ((batch_df['Month'] == 11) & (batch_df['DayofMonth'] >= 20))
        ).astype(int)

        batch_df['AirlineHistDelayRate'] = batch_df['Marketing_Airline_Network'].map(airline_hist_map).fillna(global_delay_rate)
        batch_df['OriginHistDelayRate']  = batch_df['Origin'].map(origin_hist_map).fillna(global_delay_rate)
        batch_df['AvgOriginTaxiOut']     = batch_df['Origin'].map(avg_taxiout_map).fillna(global_taxiout)
        batch_df['OriginDailyFlights']   = batch_df['Origin'].map(avg_daily_flights_map).fillna(global_daily_flights)

        for col, period in [('DepHour', 24), ('Month', 12), ('DayOfWeek', 7)]:
            batch_df[f'{col}_sin'] = np.sin(2 * np.pi * batch_df[col] / period)
            batch_df[f'{col}_cos'] = np.cos(2 * np.pi * batch_df[col] / period)

        bins   = [0, 300, 600, 1000, 1500, 9999]
        labels_db = ['Short', 'Medium', 'Long', 'X-Long', 'Ultra']
        batch_df['DistanceBucket'] = pd.cut(batch_df['Distance'], bins=bins, labels=labels_db).astype(str)

        all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        X_batch  = batch_df[[c for c in all_cols if c in batch_df.columns]]

        preds  = pipeline.predict(X_batch)
        probas = pipeline.predict_proba(X_batch)[:, 1]

        batch_df['PredictedDelay']   = preds
        batch_df['DelayProbability'] = probas.round(3)
        batch_df['DelayLabel']       = batch_df['PredictedDelay'].map({0: 'On-Time', 1: 'Delayed'})

        st.success(f"Scored {len(batch_df):,} flights.")
        st.dataframe(
            batch_df[['FlightDate', 'Marketing_Airline_Network',
                       'Origin', 'Dest', 'CRSDepTime',
                       'DelayLabel', 'DelayProbability']].head(50),
            use_container_width=True
        )

        csv_out = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_out,
            file_name="flight_delay_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
