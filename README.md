## Bitcoin Price Prediction Using LSTM (Web Application)

This project presents an end-to-end system for predicting future Bitcoin closing prices using a Long Short-Term Memory (LSTM) neural network. The work was completed as part of CSCI 6444 – Introduction to Big Data and Analytics and demonstrates the complete pipeline of data collection, preprocessing, model development, evaluation, and deployment as a web-based application.

The trained model is deployed using Flask and hosted online, allowing users to interact with the model through a browser interface without requiring any local setup.

## Live Web Application

The deployed web application can be accessed here:

👉 https://web-production-21d7.up.railway.app/

The application is publicly accessible and does not require login credentials or installation.


## 📌 Project Description

Bitcoin prices are highly volatile and influenced by market sentiment, global events, and investor behavior. These price movements are nonlinear and often unpredictable, making forecasting difficult using traditional statistical models. This project investigates whether historical Bitcoin price data can be leveraged to predict future closing prices using a deep-learning approach.

To address this problem, a Long Short-Term Memory (LSTM) neural network is used due to its ability to learn long-term dependencies in time-series data. The trained model is evaluated using standard regression metrics and then deployed as an interactive web application for real-time exploration of predictions and results.


## 📊 Data Source and Preprocessing

Data Source: Yahoo Finance (BTC-USD)
https://finance.yahoo.com/quote/BTC-USD/

Full Dataset Period: January 1, 2018 – November 16, 2025

Selected Modeling Period: August 31, 2021 – November 16, 2025

Total Records Used: 1,539 daily observations

## Preprocessing Steps

1. Selected only Date and Close price columns

2. Cleaned column names and removed inconsistencies

3. Converted dates to datetime format and sorted chronologically

4. Scaled closing prices using MinMaxScaler

5. Created 60-day sliding window sequences for supervised learning

6. Split data into 80% training and 20% testing sets


## 🧠 Model Architecture

The forecasting model is a Long Short-Term Memory (LSTM) neural network designed for time-series prediction.

Architecture Overview:

1. Input layer with shape (60 time steps, 1 feature)

2. First LSTM layer (50 units, returns sequences)

3. Dropout layer (rate = 0.2)

4. Second LSTM layer (30 units)

5. Dense output layer with one neuron

## Training Configuration:

1. Loss Function: Mean Squared Error (MSE)

2. Optimizer: Adam

3. Batch Size: 32

4. Maximum Epochs: 200

5. Early stopping applied to prevent overfitting


## 📈 Model Evaluation and Results

The model was evaluated using both training and testing datasets. Performance was measured using:

1. Root Mean Squared Error (RMSE)

2. Mean Squared Error (MSE)

3. Mean Absolute Error (MAE)

Explained Variance

R² Score

## Key Testing Results

1. R² Score: 0.9502

2. Explained Variance: 0.9503

These results indicate that the LSTM model explains over 95% of the variance in Bitcoin closing prices and generalizes well to unseen data.

## 🔮 Future Price Forecasting

After evaluation, the trained LSTM model was used to generate a 90-day future forecast beyond the last available date. The forecast was produced iteratively by feeding each predicted value back into the 60-day input sequence.

Observations:

1. Forecast trends are smooth and stable

2. No unrealistic spikes are observed

3. Predictions align with recent market behavior

## 🔍 Comparison with Traditional Methods

The LSTM approach was compared with commonly used time-series forecasting techniques:

1. Moving Average (MA)

2. Linear Regression Trend Model

3. ARIMA

Traditional models struggled with Bitcoin’s volatility and nonlinear patterns. The LSTM model significantly outperformed these methods due to its ability to learn long-term temporal dependencies and adapt to rapid market changes.

## 🛠️ Technology Stack

1. Programming Language: Python

2. Web Framework: Flask

3. Deep Learning Library: TensorFlow / Keras

4. Data Processing: NumPy, Pandas

5. Scaling: scikit-learn (MinMaxScaler)

6. Visualization: Plotly, Matplotlib

7. Deployment Platform: Railway
