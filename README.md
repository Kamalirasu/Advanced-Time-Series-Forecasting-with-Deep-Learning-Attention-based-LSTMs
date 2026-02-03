# Advanced Time Series Forecasting with Deep Learning  
## Attention-based LSTM for Multivariate Forecasting

## Project Overview

This project implements an **advanced deep learning framework for multivariate time series forecasting**, with a specific focus on integrating an **attention mechanism into a Long Short-Term Memory (LSTM) network**.  

The objective is to **outperform traditional forecasting benchmarks** such as:
- Standard LSTM
- Classical statistical models (SARIMA)

The model is evaluated on a **complex, real-world, publicly available dataset** involving **hourly energy consumption**, demonstrating robust preprocessing, model design, and rigorous experimental evaluation.

## Key Objectives

- Handle **complex multivariate time-series data**
- Implement **custom attention-based LSTM architecture**
- Compare deep learning models against **statistical benchmarks**
- Apply **robust preprocessing and feature engineering**
- Evaluate models using **industry-standard metrics**
- Interpret **attention weights** for model explainability

## Dataset

- **Source:** Public household energy consumption dataset  
- **Frequency:** Hourly  
- **Features:** Multiple continuous variables related to power usage  
- **Target Variable:** `Global_active_power`

### Preprocessing Steps:
- Missing value handling via **time-based interpolation**
- Feature scaling using **Min-Max normalization**
- Temporal feature engineering:
  - Hour of day
  - Day of week
  - Month

## Model Architectures

### 1. Attention-based LSTM (Primary Model)
- LSTM with `return_sequences=True`
- Custom **Bahdanau-style attention mechanism**
- Dense output layer for regression
- Early stopping for regularization

### 2. Standard LSTM (Benchmark)
- Single-layer LSTM
- Dense output layer
- No attention mechanism

### 3. SARIMA (Benchmark)
- Seasonal ARIMA with daily seasonality
- Classical statistical time series forecasting approach

## Attention Mechanism

The attention layer learns to assign **dynamic importance weights** to different time steps within each input sequence. This enables the model to:
- Focus on the most relevant historical information
- Improve long-horizon forecasting accuracy
- Provide interpretability via attention weight visualization

## Training Strategy

- Train/Test split: **80% / 20%**
- Sliding window sequence length: **24 hours**
- Optimizer: **Adam**
- Loss Function: **Mean Squared Error (MSE)**
- Early stopping with validation monitoring

## Evaluation Metrics

All models are evaluated using:

- **RMSE** – Root Mean Squared Error  
- **MAE** – Mean Absolute Error  
- **MAPE** – Mean Absolute Percentage Error  

Performance comparison demonstrates the effectiveness of the attention mechanism over baseline approaches.

## Results & Visualization

- Forecast comparison plots between:
  - Actual values
  - Attention-based LSTM
  - Standard LSTM
- Attention weight plots highlighting temporal importance
- Quantitative metric comparison across all models


