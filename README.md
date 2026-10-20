# Hospital Cash-Flow Predictor

## Overview
A Python-based dashboard for analyzing hospital insurance claims,
monitoring pending claims, and identifying claims that may experience
long processing times.

## Features
- Insurance claim status analysis
- Pending claim ageing analysis
- Outstanding claim amount monitoring
- Insurance-provider comparisons
- Machine learning delay classification
- Interactive charts and claim filtering
- CSV export

## Technologies
- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Scikit-learn

## Machine Learning
Compares Logistic Regression and Random Forest models to classify
whether a paid claim took more than 30 days to process.

## Dataset
The project uses 1,500 synthetically generated insurance claims.
It does not contain real hospital or patient records.

## Run the Project
Install dependencies:

pip install -r requirements.txt

Launch the dashboard:

python -m streamlit run app.py

## Limitations
The results are based on synthetic data and should not be interpreted
as validated predictions of real hospital cash flow."# hospital-cash-flow-predictor" 
