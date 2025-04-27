# Solar and Load Forecasting using CNN

This project implements a Convolutional Neural Network (CNN) based forecasting system for solar energy generation and electricity load. It uses time-series data to predict future solar generation and load demand.

## Features

- **Solar Generation Forecasting**: Predicts solar energy production based on weather patterns and historical data
- **Load Forecasting**: Predicts electricity demand based on consumption patterns and external factors
- **CNN Architecture**: Utilizes convolutional neural networks for effective time-series forecasting
- **Synthetic Data Generation**: Includes functionality to generate realistic synthetic data for demonstration
- **Comprehensive Evaluation**: Provides multiple metrics (RMSE, MAE, R², MAPE) to assess forecast accuracy
- **Visualization**: Creates plots of actual vs. predicted values for easy interpretation

## Requirements

```
numpy
pandas
matplotlib
tensorflow
scikit-learn
```

## Usage

1. Run the main script:

```bash
python solar_load_forecasting.py
```

2. The script will:
   - Generate synthetic data (or you can use your own data by modifying the code)
   - Preprocess the data for the CNN model
   - Train separate models for solar generation and load forecasting
   - Evaluate the models using multiple metrics
   - Generate forecast visualizations
   - Save results to the 'results' directory

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout for regularization
- Dense layers for final prediction

## Data Preprocessing

The system preprocesses time-series data by:
- Scaling features using MinMaxScaler
- Creating sequence-based inputs (using 24 hours to predict the next 24 hours)
- Adding time-based features (hour, day of week, month)
- Splitting data into training and testing sets

## Evaluation Metrics

The forecasts are evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Customization

You can customize the model by adjusting parameters in the main function:
- `seq_length`: Number of time steps to use as input
- `forecast_horizon`: Number of time steps to predict
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

For using your own data, replace the synthetic data generation with your data loading code and ensure it has similar features (temperature, cloud cover, solar irradiance, etc.).