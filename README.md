# Air Quality — Benzene Concentration Predictor
### CSAI-801 Artificial Intelligence & Machine Learning


---

## About the Project

Air pollution kills approximately 7 million people per year according to the WHO. 
Benzene (C6H6) is one of the most dangerous pollutants, causing respiratory 
illnesses and cancer. Precise air quality monitors cost between $20,000 and $50,000, 
making wide deployment difficult for cities.

This project proves that machine learning can predict benzene concentration 
accurately using only cheap metal oxide sensors that cost between $100 and $500 — 
no expensive reference analyzer required.

---

## Live Demo

https://benzene-predictor-zy4vymrelnxtkpva7pvf68.streamlit.app/

---

## Dataset

UCI Air Quality Dataset — 9,357 hourly readings collected between March 2004 
and April 2005 in an Italian city.

Source: https://archive.ics.uci.edu/dataset/360/air+quality

---

## Models Trained

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression (Baseline) | 0.9007 | 1.79 |
| Random Forest | 0.8398 | 2.08 |
| Gradient Boosting | 0.9074 | 1.39 |
| SVR (RBF kernel) | 0.9228 | 1.31 |

Best model: **Support Vector Regression (RBF kernel)** with R² = 0.9228

---

## Features Used

- PT08.S1 — CO metal oxide sensor signal
- PT08.S3 — NOx metal oxide sensor signal
- PT08.S4 — NO2 metal oxide sensor signal
- PT08.S5 — O3 metal oxide sensor signal
- Temperature, Relative Humidity, Absolute Humidity
- Hour, Day of Week, Month, Weekend flag, Season

---

## Files

| File | Description |
|---|---|
| app.py | Streamlit web application |
| benzene_model.pkl | Trained SVR model + scaler + features |
| requirements.txt | Python dependencies |
| CSAI 801 Final Project.ipynb | Full project notebook |
| AirQualityUCI__.csv | Dataset |
