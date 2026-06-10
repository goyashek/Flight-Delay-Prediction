# Flight Delay Prediction

End-to-end machine learning project that predicts whether a flight will arrive late using historical flight and operational data.

## Project Overview

Flight delays create operational costs for airlines and inconvenience for passengers. This project builds a classification pipeline that predicts arrival delays before departure using features available before takeoff.

## Workflow

1. Data Loading & Quality Assessment
2. Data Cleaning & Leakage Prevention
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Development
6. Cross-Validation
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Interpretation
10. Error Analysis
11. Business Recommendations

## Dataset

The original dataset (~289 MB) is not included in this repository due to GitHub size considerations.

To reproduce the project:

1. Download the dataset from the original source.
2. Update the notebook path if required.

Example:

```text
data/
└── flight_data_2018_2024.csv
```

### Target Variable

`ArrDel15`

* 0 → On-Time Flight
* 1 → Delayed Flight (>15 minutes)

## Models Trained

* Logistic Regression
* Random Forest
* XGBoost

## Techniques Demonstrated

* Missing Value Handling
* Data Leakage Detection
* Feature Engineering
* One-Hot Encoding
* Feature Scaling
* Stratified Train-Test Split
* Cross-Validation
* Hyperparameter Tuning
* ROC-AUC Evaluation
* Confusion Matrix Analysis
* SHAP Explainability
* Error Analysis

## Key Visualizations

* Delay Distribution
* Delay Causes Analysis
* Seasonal Delay Heatmap
* Hourly Delay Patterns
* Airport-Wise Delay Analysis
* Cancellation Analysis
* Weekly Delay Trends
* Correlation Matrix
* Distance vs Delay Analysis
* Confusion Matrices
* ROC Curves
* Model Comparison
* Feature Importance

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Running the Project

```bash
git clone <repository-url>
cd flight-delay-prediction
```

1. Download the dataset.
2. Place it inside the `data/` folder.
3. Open `notebooks/flight_delay_portfolio.ipynb`.
4. Run all cells sequentially.

## Project Structure

```text
flight-delay-prediction/
│
├── data/
│   └── data_source.txt
│
├── flight_delay_portfolio.ipynb
│
├── reports/
│   └── project_report.pdf
│
├── plots/
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
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Portfolio Highlights

* End-to-End Machine Learning Pipeline
* Real-World Data Cleaning
* Feature Engineering Based on Domain Knowledge
* Multiple Model Comparison
* Hyperparameter Optimization
* Model Explainability with SHAP
* Business-Oriented Insights & Recommendations

## Future Improvements

* Time-Based Validation
* Weather Data Integration
* Flight Network Features
* Real-Time Prediction API
* Model Monitoring Dashboard

## Author

**Abhi Goyal**
M.Sc. IIT Delhi
