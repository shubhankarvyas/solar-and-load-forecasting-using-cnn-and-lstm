import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to generate synthetic data for demonstration
def generate_synthetic_data(days=365, resolution_hours=1):
    """Generate synthetic solar and load data with enhanced realism.
    
    Args:
        days: Number of days of data to generate
        resolution_hours: Time resolution in hours
        
    Returns:
        DataFrame with timestamp, weather features, solar generation, and load data
    """
    n_points = int(days * 24 / resolution_hours)
    start_date = datetime(2022, 1, 1)
    timestamps = [start_date + timedelta(hours=i*resolution_hours) for i in range(n_points)]
    
    df = pd.DataFrame(index=timestamps)
    df.index.name = 'timestamp'
    
    # Enhanced temperature patterns
    seasonal_temp = 22 + 18 * np.sin(np.linspace(0, 2*np.pi*days/365, n_points))
    daily_temp = 6 * np.sin(np.linspace(0, 2*np.pi*n_points/(24/resolution_hours), n_points))
    # Add random weather events
    weather_events = np.zeros(n_points)
    n_events = int(days/30)  # Random weather events monthly
    for _ in range(n_events):
        event_start = np.random.randint(0, n_points-48)
        event_duration = np.random.randint(24, 72)
        event_magnitude = np.random.normal(0, 5)
        weather_events[event_start:event_start+event_duration] = event_magnitude
    
    temp_noise = np.random.normal(0, 2, n_points)
    for i in range(1, n_points):
        temp_noise[i] = 0.8 * temp_noise[i-1] + 0.2 * temp_noise[i]  # Temporal correlation
    
    df['temperature'] = seasonal_temp + daily_temp + temp_noise + weather_events
    
    # Enhanced cloud cover patterns
    base_cloud = np.random.normal(0.4, 0.25, n_points)
    for i in range(1, n_points):
        base_cloud[i] = 0.85 * base_cloud[i-1] + 0.15 * base_cloud[i]  # Stronger temporal correlation
    
    # Add seasonal cloud patterns
    cloud_seasonal = 0.2 + 0.3 * np.sin(np.linspace(0, 2*np.pi*days/365, n_points) + np.pi)
    # Add random cloudy periods
    cloudy_events = np.zeros(n_points)
    n_cloudy = int(days/15)  # Cloudy events every 2 weeks average
    for _ in range(n_cloudy):
        event_start = np.random.randint(0, n_points-24)
        event_duration = np.random.randint(12, 48)
        cloudy_events[event_start:event_start+event_duration] = np.random.uniform(0.2, 0.4)
    
    df['cloud_cover'] = np.clip(0.6 * base_cloud + 0.25 * cloud_seasonal + 0.15 * cloudy_events, 0, 1)
    
    # Enhanced solar irradiance calculation
    hour_of_day = np.array([t.hour for t in df.index])
    day_of_year = np.array([t.timetuple().tm_yday for t in df.index])
    
    daylight = np.zeros(n_points)
    for i in range(n_points):
        day_length = 11 + 5 * np.sin(2*np.pi*day_of_year[i]/365 - np.pi/2)  # More seasonal variation
        sunrise = 6  # Fixed sunrise time at 6 AM
        sunset = 18   # Fixed sunset time at 6 PM
        
        if sunrise <= hour_of_day[i] <= sunset:
            # Modified sinusoidal pattern for more realistic daylight
            rel_time = (hour_of_day[i] - sunrise) / (sunset - sunrise)
            daylight[i] = np.sin(np.pi * rel_time)  # Pure sinusoidal pattern
        else:
            daylight[i] = 0  # Ensure zero generation during night hours
    
    # Enhanced cloud impact on irradiance
    cloud_impact = 1 - (0.8 * df['cloud_cover'] ** 0.7)  # Non-linear cloud impact
    df['solar_irradiance'] = 1050 * daylight * cloud_impact
    
    # Define solar generation parameters first
    max_capacity = 100  # kW
    
    # More complex temperature efficiency
    temp_effect = 1.0 - 0.005 * np.maximum(0, df['temperature'] - 25) ** 1.2  # Non-linear temperature impact
    
    # Enhanced efficiency patterns
    base_efficiency = 0.185 + 0.015 * np.sin(np.linspace(0, 2*np.pi*days/365, n_points))
    dust_effect = 1.0 - 0.02 * np.cumsum(np.ones(n_points)) / n_points  # Gradual dust accumulation
    rain_cleaning = np.zeros(n_points)
    n_rain = int(days/20)  # Rain events for cleaning
    for _ in range(n_rain):
        rain_day = np.random.randint(0, n_points-24)
        rain_cleaning[rain_day:rain_day+24] = 0.015  # Efficiency recovery after rain
    
    efficiency = base_efficiency * temp_effect * (dust_effect + rain_cleaning)
    
    # Calculate generation with strict night-time zero generation
    df['solar_generation'] = max_capacity * df['solar_irradiance'] / 1000 * efficiency
    
    # Enforce strict zero generation during nighttime
    df.loc[hour_of_day < 6, 'solar_generation'] = 0  # Force zero generation before 6 AM
    df.loc[hour_of_day >= 18, 'solar_generation'] = 0  # Force zero generation after 6 PM
    df.loc[hour_of_day == 0, 'solar_generation'] = 0  # Force zero generation at midnight
    df.loc[daylight == 0, 'solar_generation'] = 0  # Ensure zero generation during all night hours
    
    # Calculate final solar generation with noise
    noise_scale = 1.5 + 2.5 * df['cloud_cover'] * (1 - df['cloud_cover'])
    generation_noise = np.random.normal(0, noise_scale, n_points)
    generation_noise[daylight == 0] = 0  # Ensure no noise during night hours
    
    df['solar_generation'] += generation_noise
    df['solar_generation'] = df['solar_generation'].clip(0)
    
    # Final enforcement of zero generation during night hours
    df.loc[daylight == 0, 'solar_generation'] = 0
    
    # Enhanced load patterns
    base_load = 200
    
    # More complex time-of-day pattern
    tod_pattern = np.zeros(24)
    # Early morning
    tod_pattern[5:8] = np.linspace(0.3, 0.6, 3)
    # Morning peak
    tod_pattern[8:11] = np.linspace(0.7, 0.9, 3)
    # Midday
    tod_pattern[11:15] = 0.85
    # Afternoon dip
    tod_pattern[15:17] = [0.75, 0.7]
    # Evening peak
    tod_pattern[17:22] = np.concatenate([np.linspace(0.8, 1.0, 3), np.linspace(0.95, 0.7, 2)])
    # Night
    tod_pattern[22:] = np.linspace(0.5, 0.3, 2)
    tod_pattern[:5] = np.linspace(0.3, 0.3, 5)
    
    # Enhanced weekly pattern
    day_of_week = np.array([t.weekday() for t in df.index])
    weekend_factor = np.ones(n_points)
    weekend_factor[day_of_week >= 5] = 0.75
    # Friday transition
    friday_afternoon = (day_of_week == 4) & (hour_of_day >= 14)
    weekend_factor[friday_afternoon] = np.linspace(1.0, 0.8, np.sum(friday_afternoon))
    # Monday ramp-up
    monday_morning = (day_of_week == 0) & (hour_of_day < 10)
    weekend_factor[monday_morning] = np.linspace(0.8, 1.0, np.sum(monday_morning))
    
    # Calculate base load
    df['load'] = base_load
    df['load'] += 180 * np.array([tod_pattern[h % 24] for h in hour_of_day])
    
    # Enhanced temperature response
    temp_load = np.zeros(n_points)
    # Cooling load with humidity impact simulation
    cooling_base = 90 / (1 + np.exp(-(df['temperature'] - 23) / 2.5))
    humidity_factor = 1 + 0.2 * np.sin(np.linspace(0, 4*np.pi*days/365, n_points))  # Simulated humidity
    temp_load += cooling_base * humidity_factor
    
    # Heating load with wind chill simulation
    heating_base = 70 / (1 + np.exp((df['temperature'] - 16) / 2.5))
    wind_factor = 1 + 0.15 * np.random.normal(0, 1, n_points)
    for i in range(1, n_points):  # Temporal correlation in wind
        wind_factor[i] = 0.9 * wind_factor[i-1] + 0.1 * wind_factor[i]
    temp_load += heating_base * wind_factor
    
    df['load'] += temp_load
    df['load'] *= weekend_factor
    
    # Add special events (holidays, etc.)
    n_special = int(days/60)  # Special events every 2 months average
    for _ in range(n_special):
        event_start = np.random.randint(0, n_points-48)
        event_duration = np.random.randint(24, 72)
        event_factor = np.random.uniform(0.7, 1.3)
        df.iloc[event_start:event_start+event_duration, df.columns.get_loc('load')] *= event_factor
    
    # Enhanced noise with temporal correlation
    base_noise = np.random.normal(0, 0.04 * df['load'], n_points)
    for i in range(1, n_points):
        base_noise[i] = 0.7 * base_noise[i-1] + 0.3 * base_noise[i]
    df['load'] += base_noise
    
    return df

# Data preprocessing functions
def preprocess_data(df, target_col, seq_length=24, forecast_horizon=24, test_split=0.2):
    """Preprocess data for CNN model.
    
    Args:
        df: DataFrame with features and target
        target_col: Column name of the target variable to predict
        seq_length: Number of time steps to use as input sequence
        forecast_horizon: Number of time steps to predict ahead
        test_split: Fraction of data to use for testing
        
    Returns:
        X_train, y_train, X_test, y_test, scalers
    """
    # Select features and target
    features = ['temperature', 'cloud_cover', 'solar_irradiance']
    if target_col == 'solar_generation':
        # For solar forecasting, we use weather features
        X_cols = features
    else:  # target_col == 'load'
        # For load forecasting, we use weather features and solar generation
        X_cols = features + ['solar_generation']
    
    # Add time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    X_cols += ['hour', 'day_of_week', 'month']
    
    # Scale features
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    df_scaled = df.copy()
    df_scaled[X_cols] = x_scaler.fit_transform(df[X_cols])
    df_scaled[[target_col]] = y_scaler.fit_transform(df[[target_col]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(df_scaled) - seq_length - forecast_horizon + 1):
        X.append(df_scaled[X_cols].iloc[i:i+seq_length].values)
        y.append(df_scaled[target_col].iloc[i+seq_length:i+seq_length+forecast_horizon].values)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, (x_scaler, y_scaler)

# CNN model for forecasting
def build_cnn_lstm_model(input_shape, output_length):
    """Build a hybrid CNN-LSTM model for time series forecasting.
    
    Args:
        input_shape: Shape of input sequences (seq_length, n_features)
        output_length: Number of time steps to predict
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Enhanced CNN layers for better feature extraction
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same',
               input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.25),
        
        # Enhanced LSTM layers for better temporal modeling
        LSTM(256, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.25),
        
        LSTM(128, return_sequences=True, recurrent_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.25),
        
        LSTM(64),
        BatchNormalization(),
        Dropout(0.25),
        
        # Enhanced dense layers for better prediction
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.25),
        
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        BatchNormalization(),
        Dropout(0.25),
        
        Dense(output_length, activation='linear')  # Linear activation for regression
    ])
    
    # Improved learning rate schedule and optimizer settings
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=500, decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    
    # Use Huber loss for better robustness to outliers
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae', 'mse'])
    
    return model

# Evaluation function
def evaluate_forecast(y_true, y_pred, scaler=None):
    """Evaluate forecast performance using multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        scaler: Scaler used to transform the target variable, for inverse transformation
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape for inverse transform
        y_true_reshaped = y_true.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        
        # Inverse transform
        y_true = scaler.inverse_transform(y_true_reshaped).reshape(y_true.shape)
        y_pred = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred.shape)
    
    # Calculate metrics
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true.flatten() != 0
    mape = np.mean(np.abs((y_true.flatten()[mask] - y_pred.flatten()[mask]) / y_true.flatten()[mask])) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# Plotting function
def plot_forecast(y_true, y_pred, title, scaler=None):
    """Plot true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        scaler: Scaler used to transform the target variable, for inverse transformation
    """
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Take first sample for plotting
        y_true_sample = y_true[0].reshape(-1, 1)
        y_pred_sample = y_pred[0].reshape(-1, 1)
        
        # Inverse transform
        y_true_sample = scaler.inverse_transform(y_true_sample).flatten()
        y_pred_sample = scaler.inverse_transform(y_pred_sample).flatten()
    else:
        y_true_sample = y_true[0]
        y_pred_sample = y_pred[0]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_sample, label='Actual', marker='o')
    plt.plot(y_pred_sample, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

# Main function
def main():
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    print("Loading data from data.csv...")
    try:
        data = pd.read_csv('results/data.csv', index_col='timestamp', parse_dates=True)
        print(f"Data loaded successfully with {len(data)} records.")
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data...")
        data = generate_synthetic_data(days=365, resolution_hours=1)
        data.to_csv('results/data.csv')
        print("Synthetic data generated and saved to results/data.csv.")
    
    # Parameters
    seq_length = 24  # Use 24 hours of data as input
    forecast_horizon = 24  # Predict 24 hours ahead
    epochs = 50
    batch_size = 32
    
    # Train and evaluate solar generation forecasting model
    print("\nTraining solar generation forecasting model...")
    X_train_solar, y_train_solar, X_test_solar, y_test_solar, solar_scalers = preprocess_data(
        data, 'solar_generation', seq_length, forecast_horizon)
    
    print(f"Input shape: {X_train_solar.shape}, Output shape: {y_train_solar.shape}")
    
    solar_model = build_cnn_lstm_model(input_shape=(seq_length, X_train_solar.shape[2]), 
                                     output_length=forecast_horizon)
    
    # Callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('results/solar_model.h5', save_best_only=True)
    ]
    
    # Train model
    solar_history = solar_model.fit(
        X_train_solar, y_train_solar,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    solar_pred = solar_model.predict(X_test_solar)
    solar_metrics = evaluate_forecast(y_test_solar, solar_pred, solar_scalers[1])
    
    print("\nSolar Generation Forecast Metrics:")
    for metric, value in solar_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_forecast(y_test_solar, solar_pred, 'Solar Generation Forecast', solar_scalers[1])
    
    # Train and evaluate load forecasting model
    print("\nTraining load forecasting model...")
    X_train_load, y_train_load, X_test_load, y_test_load, load_scalers = preprocess_data(
        data, 'load', seq_length, forecast_horizon)
    
    print(f"Input shape: {X_train_load.shape}, Output shape: {y_train_load.shape}")
    
    load_model = build_cnn_lstm_model(input_shape=(seq_length, X_train_load.shape[2]), 
                                   output_length=forecast_horizon)
    
    # Callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('results/load_model.h5', save_best_only=True)
    ]
    
    # Train model
    load_history = load_model.fit(
        X_train_load, y_train_load,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    load_pred = load_model.predict(X_test_load)
    load_metrics = evaluate_forecast(y_test_load, load_pred, load_scalers[1])
    
    print("\nLoad Forecast Metrics:")
    for metric, value in load_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    plot_forecast(y_test_load, load_pred, 'Load Forecast', load_scalers[1])
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(solar_history.history['loss'], label='Train')
    plt.plot(solar_history.history['val_loss'], label='Validation')
    plt.title('Solar Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(load_history.history['loss'], label='Train')
    plt.plot(load_history.history['val_loss'], label='Validation')
    plt.title('Load Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    print("\nForecasting complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()