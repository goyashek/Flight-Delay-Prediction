# ✈️ Flight Delay Prediction

Predicts whether a US domestic flight will arrive more than 15 minutes late — using only information available before the flight takes off.

Built as an end-to-end ML project: raw data → cleaning → feature engineering → model training → interactive Streamlit app.

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/goyashek/Flight-Delay-Prediction
cd Flight-Delay-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### 3. Get the dataset

The CSV (~289 MB) isn't included due to GitHub size limits.  
Download it and place it here:

```
data/
└── flight_data_2018_2024.csv
```

### 4. Run the notebooks

Open and run both notebooks in order:

| Notebook | What it does |
|---|---|
| `notebooks/EDA.ipynb` | Exploratory analysis and visualizations |
| `notebooks/Core-ML.ipynb` | Feature engineering, model training, evaluation |

### 5. Export the model

Open `notebooks/pickle extract.ipynb` and run all cells.  
This saves 3 files into the `streamlit/` folder:

```
streamlit/
├── flight_delay_model.pkl      ← trained XGBoost pipeline
├── flight_delay_lookups.pkl    ← dropdown values for the UI
└── flight_delay_stats.pkl      ← historical delay rates used as features
```

### 6. Launch the Streamlit app

```bash
cd streamlit
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.  
Fill in the flight details and hit **Predict** to see the result.

---

## Project Structure

```
Flight-Delay-Prediction/
│
├── data/
│   └── flight_data_2018_2024.csv       ← not tracked in git
│
├── notebooks/
│   ├── EDA.ipynb                        ← exploratory analysis
│   ├── Core-ML.ipynb                    ← model training pipeline
│   └── pickle extract.ipynb            ← exports model + pickle files
│
├── plots/                               ← all generated visualizations
│   ├── plot1_delay_distribution.png
│   ├── plot2_delay_causes.png
│   ├── plot3_seasonal_heatmap.png
│   ├── plot4_hourly_pattern.png
│   ├── plot5_airport_delay.png
│   ├── plot6_cancellations.png
│   ├── plot7_weekly_pattern.png
│   ├── plot8_correlation.png
│   ├── plot9_distance_bucket.png
│   ├── plot10_confusion_matrices.png
│   ├── plot11_roc_curves.png
│   ├── plot12_model_comparison.png
│   └── plot13_feature_importance.png
│
├── streamlit/
│   ├── app.py                           ← Streamlit prediction app
│   ├── flight_delay_model.pkl
│   ├── flight_delay_lookups.pkl
│   └── flight_delay_stats.pkl
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## What it predicts

**Target:** `ArrDel15` — did the flight arrive more than 15 minutes late?

- `0` → On-Time
- `1` → Delayed

About **24%** of non-cancelled flights in the dataset are delayed by this definition.

---

## Models

Three classifiers were trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, fast and interpretable |
| Random Forest | Strong performance, good feature importance |
| XGBoost | Best overall — used in the app |

All models use a sklearn `Pipeline` with median imputation, standard scaling, and one-hot encoding — so there's no data leakage between train and test.

---

## Key Features Used

All features are knowable **before departure** (no leakage):

- Scheduled departure/arrival time, flight duration, distance
- Airline, origin airport, destination airport
- Day of week, month, hour of day (with cyclic encoding)
- Peak hour flag, weekend flag, holiday window flag
- Historical delay rate per airline and origin airport
- Average taxi-out time at origin (congestion proxy)

---

## Author

**Abhi Goyal** · M.Sc. IIT Delhi  
[github.com/goyashek](https://github.com/goyashek)
