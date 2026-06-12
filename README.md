# ✈️ Flight Delay Prediction

An end-to-end machine learning project that predicts whether a US domestic flight will arrive more than 15 minutes late, using only information available before takeoff.

From raw data processing and feature engineering to model training and deployment, this project culminates in an interactive web application.

**[🌐 Try the Live App](https://flight-delay-xgb.streamlit.app/)**

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/goyashek/Flight-Delay-Prediction
cd Flight-Delay-Prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### 3. Get the dataset
The raw CSV (~289 MB) is not included due to GitHub's file size limits. 
Download the data by following the instructions in `data/data.txt` and place the CSV in the data folder:
```text
data/
└── flight_data_2018_2024.csv
```

### 4. Run the notebooks
Open and run the Jupyter notebooks in sequential order to explore the data and train the models:

| Notebook | Description |
|---|---|
| `notebooks/EDA.ipynb` | Exploratory data analysis and visualizations |
| `notebooks/Core-ML.ipynb` | Feature engineering, model training, and evaluation |

### 5. Export the model
To generate the required files for the Streamlit app, open and run all cells in `notebooks/pickle extract.ipynb`. This saves three files into the `streamlit/` folder:
* `flight_delay_model.pkl` — The trained XGBoost pipeline
* `flight_delay_lookups.pkl` — Dropdown values for the UI
* `flight_delay_stats.pkl` — Historical delay rates used as features

*(Note: Pre-exported files are already included in the `streamlit/` folder, so you can skip this step if you just want to run the app right away).*

### 6. Launch the Streamlit app
```bash
cd streamlit
streamlit run app.py
```
Open `http://localhost:8501` in your browser, fill in the flight details, and hit **Predict** to see the result.

---

## 📂 Project Structure

```text
Flight-Delay-Prediction/
│
├── data/
│   └── flight_data_2018_2024.csv        ← not tracked in git
│
├── notebooks/
│   ├── EDA.ipynb                        ← exploratory analysis
│   ├── Core-ML.ipynb                    ← model training pipeline
│   └── pickle extract.ipynb             ← exports model + pickle files
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

## 🎯 What it Predicts

**Target Variable:** `ArrDel15` — Did the flight arrive more than 15 minutes late?

* `0` → On-Time
* `1` → Delayed

*(Note: About **24%** of non-cancelled flights in the dataset are delayed by this definition).*

---

## 🧠 Models

Six classifiers were trained and compared. All models use a scikit-learn `Pipeline` handling median imputation, standard scaling, and one-hot encoding—ensuring there is absolutely no data leakage between the train and test sets.


| Model               | Threshold      | Accuracy   | Precision  | Recall     | F1 Score   | ROC-AUC    | Notes                                            |
| ------------------- | -------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------------------------------------ |
| Logistic Regression | Default        | 0.6286     | 0.3375     | 0.6130     | 0.4353     | 0.6729     | Baseline, fast, and highly interpretable         |
| Random Forest       | Default        | 0.6988     | 0.4062     | 0.6276     | 0.4932     | 0.7407     | Strong performance with clear feature importance |
| XGBoost             | Default        | 0.7248     | 0.4405     | 0.6604     | 0.5284     | **0.7732** | Best tree-based model — deployed in the app      |
| Neural Net          | 0.50           | 0.7140     | 0.4293     | **0.6820** | 0.5269     | 0.7724     | Highest recall among tested models               |
| Neural Net          | 0.56 (Best F1) | 0.7460     | 0.4668     | 0.6160     | **0.5311** | 0.7724     | Best F1 score overall                            |
| Neural Net          | 0.75           | **0.7989** | **0.6077** | 0.3914     | 0.4761     | 0.7724     | Highest accuracy and precision                   |



---

## 📊 Key Features Used

To ensure realistic predictions, all features rely strictly on data available **before departure**:

* Scheduled departure/arrival times, flight duration, and distance
* Airline, origin airport, and destination airport
* Temporal features: Day of week, month, and hour of day (using cyclic encoding)
* Contextual flags: Peak hour, weekend, and holiday window
* Historical delay rates per airline and origin airport
* Average taxi-out time at the origin (acting as a proxy for airport congestion)

---

## 👤 Author

**Abhishek Goyal** · M.Sc. IIT Delhi  
[github.com/goyashek](https://github.com/goyashek)
