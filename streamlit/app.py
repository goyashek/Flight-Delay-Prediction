"""
Run:
    pip install streamlit scikit-learn xgboost pandas numpy
    streamlit run app.py

Make sure these 3 files are in the same folder as app.py:
    flight_delay_model.pkl
    flight_delay_lookups.pkl
    flight_delay_stats.pkl
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st


from pathlib import Path

BASE_DIR = Path(__file__).parent   # ← always points to the streamlit/ folder

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / 'flight_delay_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(BASE_DIR / 'flight_delay_lookups.pkl', 'rb') as f:
        lookups = pickle.load(f)
    with open(BASE_DIR / 'flight_delay_stats.pkl', 'rb') as f:
        stats = pickle.load(f)
    return model, lookups, stats

model, lookups, stats = load_artifacts()


st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️")
st.title("✈️ Flight Delay Predictor")
st.write("Fill in the flight details below to check if the flight is likely to be delayed (>15 min late).")
st.divider()

col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", options=lookups['airlines'])
    origin  = st.selectbox("Origin Airport", options=lookups['origins'])
    dest    = st.selectbox("Destination Airport", options=lookups['dests'])
    dep_block = st.selectbox(
        "Departure Time Block",
        options=lookups['dep_blocks'],
        help="e.g.  0600-0659  means departing between 6:00 and 6:59 AM"
    )

with col2:
    flight_date = st.date_input("Flight Date")
    dep_time    = st.slider("Scheduled Departure (HHMM)", 0, 2359, 800, step=5,
                            help="e.g. 800 = 8:00 AM, 1430 = 2:30 PM")
    arr_time    = st.slider("Scheduled Arrival (HHMM)",  0, 2359, 1000, step=5)
    distance    = st.number_input("Distance (miles)", min_value=50, max_value=5000,
                                  value=800, step=50)
    elapsed     = st.number_input("Scheduled Flight Time (minutes)",
                                  min_value=20, max_value=800, value=120, step=5)

st.divider()
predict_btn = st.button("🔍 Predict", use_container_width=True)


def build_features(airline, origin, dest, dep_block,
                   flight_date, dep_time, arr_time, distance, elapsed):

    month       = flight_date.month
    day_of_month = flight_date.day
    day_of_week = flight_date.weekday()   
    dep_hour    = dep_time // 100

    # engineered flags
    is_weekend   = int(day_of_week in [5, 6])
    is_peak_hour = int(dep_hour in list(range(7, 10)) + list(range(16, 20)))

    season_map = {12:'Winter',1:'Winter',2:'Winter',
                   3:'Spring',4:'Spring',5:'Spring',
                   6:'Summer',7:'Summer',8:'Summer',
                   9:'Fall',10:'Fall',11:'Fall'}
    season = season_map[month]

    holiday_window = int(
        (month == 12 and day_of_month >= 15) or
        (month ==  1 and day_of_month <=  7) or
        (month ==  6 and day_of_month >= 15) or
        (month ==  7 and day_of_month <=  7) or
        (month == 11 and day_of_month >= 20)
    )

    # distance bucket
    bins   = [0, 300, 600, 1000, 1500, 9999]
    labels = ['Short', 'Medium', 'Long', 'X-Long', 'Ultra']
    dist_bucket = labels[next(i for i, b in enumerate(bins[1:]) if distance <= b)]

    # historical stats (fallback to mean if unseen value)
    airline_hist  = stats['airline_delay_rate'].get(airline,  stats['mean_airline_delay_rate'])
    origin_hist   = stats['origin_delay_rate'].get(origin,    stats['mean_origin_delay_rate'])
    avg_taxiout   = stats['avg_taxiout'].get(origin,          stats['mean_taxiout'])
    daily_flights = stats['origin_daily_flights'].get(origin, stats['mean_daily_flights'])

    # cyclic encoding
    dep_h_sin = np.sin(2 * np.pi * dep_hour   / 24)
    dep_h_cos = np.cos(2 * np.pi * dep_hour   / 24)
    m_sin     = np.sin(2 * np.pi * month       / 12)
    m_cos     = np.cos(2 * np.pi * month       / 12)
    dow_sin   = np.sin(2 * np.pi * day_of_week /  7)
    dow_cos   = np.cos(2 * np.pi * day_of_week /  7)

    row = {
        # numeric features (same order as NUMERIC_FEATURES in notebook)
        'CRSDepTime':         dep_time,
        'CRSArrTime':         arr_time,
        'CRSElapsedTime':     elapsed,
        'Distance':           distance,
        'Month':              month,
        'DayofMonth':         day_of_month,
        'DayOfWeek':          day_of_week,
        'DepHour':            dep_hour,
        'IsWeekend':          is_weekend,
        'IsPeakHour':         is_peak_hour,
        'HolidayWindow':      holiday_window,
        'OriginDailyFlights': daily_flights,
        'AirlineHistDelayRate': airline_hist,
        'OriginHistDelayRate':  origin_hist,
        'AvgOriginTaxiOut':   avg_taxiout,
        'DepHour_sin':        dep_h_sin,
        'DepHour_cos':        dep_h_cos,
        'Month_sin':          m_sin,
        'Month_cos':          m_cos,
        'DayOfWeek_sin':      dow_sin,
        'DayOfWeek_cos':      dow_cos,
        # categorical features
        'Marketing_Airline_Network': airline,
        'Origin':             origin,
        'Dest':               dest,
        'DistanceBucket':     dist_bucket,
        'Season':             season,
        'DepTimeBlk':         dep_block,
    }
    return pd.DataFrame([row])

# ── Prediction ───────────────────────────────────────────────────────────────

if predict_btn:
    X_input = build_features(airline, origin, dest, dep_block,
                             flight_date, dep_time, arr_time, distance, elapsed)

    prob    = model.predict_proba(X_input)[0][1]   # probability of delay
    delayed = prob >= 0.5

    st.subheader("Result")

    if delayed:
        st.error(f" **Likely Delayed** — {prob*100:.1f}% chance of arriving >15 min late")
    else:
        st.success(f" **Likely On-Time** — only {prob*100:.1f}% chance of delay")

    # confidence bar
    st.progress(float(prob), text=f"Delay probability: {prob*100:.1f}%")

    # quick summary of key risk factors
    dep_hour_val = dep_time // 100
    with st.expander("factors checked:: "):
        st.write(f"- **Departure hour**: {dep_hour_val}:00  "
                 f"({'peak hour 🔴' if dep_hour_val in list(range(7,10))+list(range(16,20)) else 'off-peak 🟢'})")
        st.write(f"- **Weekend flight**: {'Yes 🔴' if flight_date.weekday() >= 5 else 'No 🟢'}")
        st.write(f"- **Holiday window**: {'Yes 🔴' if (flight_date.month==12 and flight_date.day>=15) else 'No 🟢'}")
        st.write(f"- **Airline hist. delay rate**: {stats['airline_delay_rate'].get(airline, stats['mean_airline_delay_rate'])*100:.1f}%")
        st.write(f"- **Origin airport delay rate**: {stats['origin_delay_rate'].get(origin, stats['mean_origin_delay_rate'])*100:.1f}%")

st.divider()
st.caption("Model: XGBoost trained on US domestic flight data 2018-2024 · Target: arrival delay > 15 min")
