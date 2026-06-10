# Flight Delay Prediction

End-to-end machine learning project that predicts whether a flight will arrive late using historical flight and operational data.

## Project Overview

Flight delays create operational costs for airlines and inconvenience for passengers. This project builds a classification pipeline that predicts arrival delays before departure using features available at scheduling time.

## Workflow

1. Data loading and quality assessment
2. Data cleaning and leakage prevention
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Model development
6. Cross-validation
7. Hyperparameter tuning
8. Model evaluation
9. Model interpretation (Feature Importance + SHAP)
10. Error analysis
11. Business recommendations

## Dataset Features

Examples of variables used:

- Airline
- Origin airport
- Destination airport
- Scheduled departure time
- Day of week
- Month
- Route information
- Distance
- Historical airline performance

Target:

- `IsDelayed` (binary classification)

## Models Trained

- Logistic Regression
- Random Forest
- XGBoost

## Techniques Demonstrated

- Missing value handling
- Data leakage detection
- Feature engineering
- One-hot encoding
- Scaling
- Stratified train-test split
- Cross-validation
- Hyperparameter tuning
- ROC-AUC evaluation
- Confusion matrix analysis
- SHAP explainability
- Error analysis

## Key Visualizations

- Delay distribution
- Delay causes by airline
- Seasonal delay heatmap
- Delay rate by departure hour
- Airport-wise delay analysis
- Cancellation analysis
- Weekly delay trends
- Correlation matrix
- Distance vs delay analysis

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

## Running the Project

1. Place the dataset in the project directory.
2. Update the dataset path if needed.
3. Run the notebook from top to bottom.

## Project Structure

```text
├── flight_delay_portfolio_cleaned.ipynb
├── README.md
└── flight_delay_sample.csv
```

## Portfolio Highlights

This project demonstrates:

- End-to-end ML workflow
- Feature engineering based on domain knowledge
- Multiple model comparison
- Hyperparameter optimization
- Model interpretability
- Business-focused insights and recommendations

## Future Improvements

- Time-based validation
- Weather integration
- Flight network features
- Real-time prediction API
- Model monitoring dashboard

## Author

Abhi Goyal
M.Sc. IIT Delhi
