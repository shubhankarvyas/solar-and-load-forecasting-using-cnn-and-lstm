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
    
    # Enhanced solar irradiance calculation with seasonal variations
    hour_of_day = np.array([t.hour for t in df.index])
    day_of_year = np.array([t.timetuple().tm_yday for t in df.index])
    latitude = 35.0  # Example latitude (adjust as needed)
    
    daylight = np.zeros(n_points)
    for i in range(n_points):
        # Calculate sunrise and sunset times based on day of year
        B = 2 * np.pi * (day_of_year[i] - 81) / 365
        EoT = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)  # Equation of Time
        declination = 23.45 * np.sin(2 * np.pi * (day_of_year[i] + 284) / 365)  # Solar declination
        
        # Calculate sunrise and sunset hours
        cos_omega = -np.tan(np.radians(latitude)) * np.tan(np.radians(declination))
        cos_omega = np.clip(cos_omega, -1, 1)  # Ensure valid range for arccos
        omega = np.degrees(np.arccos(cos_omega))
        sunrise = 12 - omega/15 - EoT/60
        sunset = 12 + omega/15 - EoT/60
        day_length = sunset - sunrise
        
        if sunrise <= hour_of_day[i] <= sunset:
            solar_elevation = np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) + \
                             np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * \
                             np.cos(np.radians(15 * (hour_of_day[i] - 12)))
            daylight[i] = np.maximum(0, solar_elevation)  # Realistic solar elevation pattern
        else:
            daylight[i] = 0  # Night time
    
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
        X_train, y_train, X_test, y_test, scalers, test_timestamps
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
    timestamps = []  # To store timestamps for each sequence
    
    for i in range(len(df_scaled) - seq_length - forecast_horizon + 1):
        X.append(df_scaled[X_cols].iloc[i:i+seq_length].values)
        y.append(df_scaled[target_col].iloc[i+seq_length:i+seq_length+forecast_horizon].values)
        # Store the timestamps for this forecast period
        timestamps.append(df.index[i+seq_length:i+seq_length+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets ensuring representation from all months
    df_months = df.index.month.unique()
    X_train, X_test = [], []
    y_train, y_test = [], []
    test_timestamps_list = []
    
    for month in df_months:
        month_indices = [i for i, ts in enumerate(timestamps) if ts[0].month == month]
        month_train_size = int(len(month_indices) * (1 - test_split))
        
        # Split indices for this month
        train_indices = month_indices[:month_train_size]
        test_indices = month_indices[month_train_size:]
        
        # Add to training and test sets
        X_train.extend([X[i] for i in train_indices])
        X_test.extend([X[i] for i in test_indices])
        y_train.extend([y[i] for i in train_indices])
        y_test.extend([y[i] for i in test_indices])
        test_timestamps_list.extend([timestamps[i] for i in test_indices])
    
    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    test_timestamps = test_timestamps_list
    
    return X_train, y_train, X_test, y_test, (x_scaler, y_scaler), test_timestamps

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

# Function to apply physical constraints to predictions
def apply_solar_constraints(y_pred, timestamps):
    """Apply physical constraints to solar generation predictions.
    
    Args:
        y_pred: Predicted values (scaled)
        timestamps: List of timestamps for each prediction sequence
        
    Returns:
        Constrained predictions
    """
    y_pred_constrained = y_pred.copy()
    
    for i in range(len(y_pred)):
        # Get hours for this prediction sequence
        hours = np.array([ts.hour for ts in timestamps[i]])
        
        # Apply nighttime constraints
        night_mask = (hours < 6) | (hours >= 18)
        y_pred_constrained[i, night_mask] = 0.0
    
    return y_pred_constrained

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
    mape = np.mean(np.abs((y_true.flatten()[mask] - y_pred.flatten()[mask]) / y_true.flatten()[mask])) * 100 if np.any(mask) else np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# Plotting function
# Modified plotting function for daily forecasts
def plot_forecast(y_true, y_pred, title, timestamps=None, scaler=None, sample_idx=0):
    """Plot true vs predicted values with proper hour formatting.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        timestamps: List of timestamps for the prediction
        scaler: Scaler used to transform the target variable, for inverse transformation
        sample_idx: Index of the sample to plot
    """
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Take specified sample for plotting
        y_true_sample = y_true[sample_idx].reshape(-1, 1)
        y_pred_sample = y_pred[sample_idx].reshape(-1, 1)
        
        # Inverse transform
        y_true_sample = scaler.inverse_transform(y_true_sample).flatten()
        y_pred_sample = scaler.inverse_transform(y_pred_sample).flatten()
    else:
        y_true_sample = y_true[sample_idx]
        y_pred_sample = y_pred[sample_idx]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Use hour labels on x-axis if timestamps are provided
    if timestamps is not None:
        hours = [ts.hour for ts in timestamps[sample_idx]]
        
        # Plot with proper continuous lines
        plt.plot(range(len(hours)), y_true_sample, label='Actual', marker='o', linestyle='-')
        plt.plot(range(len(hours)), y_pred_sample, label='Predicted', marker='x', linestyle='-')
        
        # Fix x-axis ticks to show only valid hours (0-23)
        plt.xticks(range(len(hours)), hours)
        plt.xlim(-0.5, len(hours)-0.5)  # Ensure proper bounds
        plt.xlabel('Hour of Day')
        
        # Add vertical lines for sunrise and sunset times (approximately)
        for i, hour in enumerate(hours):
            if hour < 6 or hour >= 18:
                plt.axvspan(i-0.5, i+0.5, alpha=0.2, color='gray')
        
        # Add night label only once in legend
        plt.axvspan(-10, -9, alpha=0.2, color='gray', label='Night')  # Off-screen for legend only
    else:
        plt.plot(y_true_sample, label='Actual', marker='o', linestyle='-')
        plt.plot(y_pred_sample, label='Predicted', marker='x', linestyle='-')
        plt.xlabel('Time Steps')
    
    plt.title(title)
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Ensure plots are saved to results directory
    plt.savefig(f'results/{title.replace(" ", "_")}.png')
    plt.close()
# Add the missing plot_daily_pattern function
def plot_daily_pattern(df, column, title):
    """Plot the average daily pattern of a variable.
    
    Args:
        df: DataFrame with time index
        column: Column name to plot
        title: Plot title
    """
    # Group by hour and calculate mean
    hourly_avg = df.groupby(df.index.hour)[column].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linestyle='-')
    plt.title(f'Average Daily {title} Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel(title)
    plt.grid(True)
    plt.xticks(range(0, 24))  # Ensure all hours 0-23 are shown
    plt.xlim(-0.5, 23.5)  # Set proper x-axis limits
    plt.savefig(f'results/daily_{title.lower().replace(" ", "_")}.png')
    plt.close()

# Also add the plot_training_history function for completeness
def plot_training_history(solar_history, load_history):
    """Plot training history for solar and load models.
    
    Args:
        solar_history: History object from solar model training
        load_history: History object from load model training
    """
    plt.figure(figsize=(12, 10))
    
    # Loss plots
    plt.subplot(2, 2, 1)
    plt.plot(solar_history.history['loss'], label='Train')
    plt.plot(solar_history.history['val_loss'], label='Validation')
    plt.title('Solar Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(load_history.history['loss'], label='Train')
    plt.plot(load_history.history['val_loss'], label='Validation')
    plt.title('Load Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # MAE plots
    plt.subplot(2, 2, 3)
    plt.plot(solar_history.history['mae'], label='Train')
    plt.plot(solar_history.history['val_mae'], label='Validation')
    plt.title('Solar Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(load_history.history['mae'], label='Train')
    plt.plot(load_history.history['val_mae'], label='Validation')
    plt.title('Load Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
# New function for annual forecast visualization
def plot_annual_forecast(data, y_true, y_pred, title, timestamps, scaler=None):
    """Plot forecast for a longer period (annual trends).
    
    Args:
        data: Original DataFrame with time index
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        timestamps: List of timestamps for each prediction
        scaler: Scaler used to transform the target variable
    """
    # Create a DataFrame to store all predictions with proper timestamps
    forecast_df = pd.DataFrame()
    
    # Get all test sequences
    for i in range(len(y_true)):
        if scaler is not None:
            # Transform to original scale
            true_vals = scaler.inverse_transform(y_true[i].reshape(-1, 1)).flatten()
            pred_vals = scaler.inverse_transform(y_pred[i].reshape(-1, 1)).flatten()
        else:
            true_vals = y_true[i]
            pred_vals = y_pred[i]
        
        # Create temporary DataFrame for this sequence
        temp_df = pd.DataFrame({
            'timestamp': timestamps[i],
            'actual': true_vals,
            'predicted': pred_vals
        })
        
        # Append to main DataFrame
        forecast_df = pd.concat([forecast_df, temp_df], ignore_index=True)
    
    # Sort by timestamp
    forecast_df = forecast_df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates - keep the first occurrence of each timestamp
    forecast_df = forecast_df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Set timestamp as index
    forecast_df = forecast_df.set_index('timestamp')
    
    # Create daily and monthly averages for smoother visualization
    daily_avg = forecast_df.resample('D').mean()
    monthly_avg = forecast_df.resample('M').mean()
    
    # Create plot
    plt.figure(figsize=(18, 10))
    
    # Plot daily averages with moderate transparency
    plt.plot(daily_avg.index, daily_avg['actual'], 'b-', alpha=0.6, linewidth=1.5, label='Actual (Daily Avg)')
    plt.plot(daily_avg.index, daily_avg['predicted'], 'r-', alpha=0.6, linewidth=1.5, label='Predicted (Daily Avg)')
    
    # Plot monthly averages with solid lines
    plt.plot(monthly_avg.index, monthly_avg['actual'], 'b-', linewidth=3, label='Actual (Monthly Avg)')
    plt.plot(monthly_avg.index, monthly_avg['predicted'], 'r-', linewidth=3, label='Predicted (Monthly Avg)')
    
    # Highlight daytime/nighttime patterns if title contains 'Solar'
    if 'Solar' in title:
        # Add shaded areas for winter/summer seasons
        plt.axvspan(pd.Timestamp('2022-12-21'), pd.Timestamp('2022-12-31'), alpha=0.1, color='lightblue', label='Winter')
        plt.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-03-20'), alpha=0.1, color='lightblue')
        plt.axvspan(pd.Timestamp('2022-06-21'), pd.Timestamp('2022-09-22'), alpha=0.1, color='lightyellow', label='Summer')
    
    # Formatting
    plt.title(f'Annual {title} Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    
    # Add year as text annotation
    plt.figtext(0.5, 0.01, '2022', ha='center', fontsize=14)
    
    # Save to results directory
    plt.savefig(f'results/annual_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_patterns(data, y_true, y_pred, timestamps, solar_scaler, load_scaler):
    """Create a visualization of seasonal patterns for both solar generation and load.
    
    Args:
        data: Original DataFrame with time index
        y_true: Dictionary with solar and load true values
        y_pred: Dictionary with solar and load predicted values
        timestamps: Dictionary with solar and load timestamps
        solar_scaler: Scaler for solar data
        load_scaler: Scaler for load data
    """
    # First, ensure we have data for all months by using the original data
    full_data = data.copy()
    full_data['month'] = full_data.index.month
    full_data['hour'] = full_data.index.hour
    
    # Create DataFrames for solar and load forecasts
    solar_df = pd.DataFrame()
    load_df = pd.DataFrame()
    
    # Process solar data
    for i in range(len(y_true['solar'])):
        true_vals = solar_scaler.inverse_transform(y_true['solar'][i].reshape(-1, 1)).flatten()
        pred_vals = solar_scaler.inverse_transform(y_pred['solar'][i].reshape(-1, 1)).flatten()
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps['solar'][i],
            'actual': true_vals,
            'predicted': pred_vals
        })
        solar_df = pd.concat([solar_df, temp_df], ignore_index=True)
    
    # Process load data
    for i in range(len(y_true['load'])):
        true_vals = load_scaler.inverse_transform(y_true['load'][i].reshape(-1, 1)).flatten()
        pred_vals = load_scaler.inverse_transform(y_pred['load'][i].reshape(-1, 1)).flatten()
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps['load'][i],
            'actual': true_vals,
            'predicted': pred_vals
        })
        load_df = pd.concat([load_df, temp_df], ignore_index=True)
    
    # Sort and clean up DataFrames
    solar_df = solar_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    load_df = load_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    
    # Add month and hour information
    solar_df['month'] = solar_df['timestamp'].dt.month
    solar_df['hour'] = solar_df['timestamp'].dt.hour
    load_df['month'] = load_df['timestamp'].dt.month
    load_df['hour'] = load_df['timestamp'].dt.hour
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    solar_df['season'] = solar_df['month'].apply(get_season)
    load_df['season'] = load_df['month'].apply(get_season)
    
    # Create the figure
    plt.figure(figsize=(20, 15))
    
    # 1. Solar Generation by Season - Daily Patterns using full data
    plt.subplot(2, 2, 1)
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = full_data[full_data['season'] == season]
        hourly_avg = season_data.groupby('hour')['solar_generation'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, linewidth=2, label=season)
    
    plt.title('Solar Generation - Daily Pattern by Season', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Generation (kW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, 24, 2))
    
    # 2. Load by Season - Daily Patterns using full data
    plt.subplot(2, 2, 2)
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = full_data[full_data['season'] == season]
        hourly_avg = season_data.groupby('hour')['load'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, linewidth=2, label=season)
    
    plt.title('Load - Daily Pattern by Season', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Load (kW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(0, 24, 2))
    
    # 3. Monthly Average Solar Generation - Actual vs Predicted
    plt.subplot(2, 2, 3)
    monthly_solar = solar_df.groupby('month').agg({'actual': 'mean', 'predicted': 'mean'})
    
    plt.plot(monthly_solar.index, monthly_solar['actual'], 'b-', marker='o', linewidth=2, label='Actual')
    plt.plot(monthly_solar.index, monthly_solar['predicted'], 'r-', marker='x', linewidth=2, label='Predicted')
    
    plt.title('Monthly Average Solar Generation', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Generation (kW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # 4. Monthly Average Load - Actual vs Predicted
    plt.subplot(2, 2, 4)
    monthly_load = load_df.groupby('month').agg({'actual': 'mean', 'predicted': 'mean'})
    
    plt.plot(monthly_load.index, monthly_load['actual'], 'b-', marker='o', linewidth=2, label='Actual')
    plt.plot(monthly_load.index, monthly_load['predicted'], 'r-', marker='x', linewidth=2, label='Predicted')
    
    plt.title('Monthly Average Load', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Load (kW)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    plt.savefig('results/seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
# Modify main function to include annual forecasts
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
    
    # Create daily pattern plots
    plot_daily_pattern(data, 'solar_generation', 'Solar Generation')
    plot_daily_pattern(data, 'load', 'Load')
    
    # Parameters
    seq_length = 24  # Use 24 hours of data as input
    forecast_horizon = 24  # Predict 24 hours ahead
    epochs = 50
    batch_size = 32
    
    # Train and evaluate solar generation forecasting model
    print("\nTraining solar generation forecasting model...")
    X_train_solar, y_train_solar, X_test_solar, y_test_solar, solar_scalers, solar_test_timestamps = preprocess_data(
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
    
    # Generate raw predictions
    solar_pred_raw = solar_model.predict(X_test_solar)
    
    # Apply physical constraints to solar predictions
    solar_pred = apply_solar_constraints(solar_pred_raw, solar_test_timestamps)
    
    # Evaluate model
    solar_metrics_raw = evaluate_forecast(y_test_solar, solar_pred_raw, solar_scalers[1])
    solar_metrics = evaluate_forecast(y_test_solar, solar_pred, solar_scalers[1])
    
    print("\nSolar Generation Forecast Metrics (Raw):")
    for metric, value in solar_metrics_raw.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nSolar Generation Forecast Metrics (With Physical Constraints):")
    for metric, value in solar_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results for multiple samples - daily forecasts
    for i in range(min(3, len(solar_test_timestamps))):
        plot_forecast(y_test_solar, solar_pred, f'Solar Generation Forecast Sample {i+1}', 
                     solar_test_timestamps, solar_scalers[1], sample_idx=i)
    
    # Plot annual solar forecast
    plot_annual_forecast(data, y_test_solar, solar_pred, 'Solar Generation', 
                        solar_test_timestamps, solar_scalers[1])
    
    # Train and evaluate load forecasting model
    print("\nTraining load forecasting model...")
    X_train_load, y_train_load, X_test_load, y_test_load, load_scalers, load_test_timestamps = preprocess_data(
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
    
    # Plot results for multiple samples - daily forecasts
    for i in range(min(3, len(load_test_timestamps))):
        plot_forecast(y_test_load, load_pred, f'Load Forecast Sample {i+1}', 
                     load_test_timestamps, load_scalers[1], sample_idx=i)
    
    # Plot annual load forecast
    plot_annual_forecast(data, y_test_load, load_pred, 'Load', 
                        load_test_timestamps, load_scalers[1])
    
    # Plot training history
    plot_training_history(solar_history, load_history)
    
    # Create seasonal patterns visualization
    # Package data for seasonal patterns plot
    y_true_dict = {'solar': y_test_solar, 'load': y_test_load}
    y_pred_dict = {'solar': solar_pred, 'load': load_pred}
    timestamps_dict = {'solar': solar_test_timestamps, 'load': load_test_timestamps}
    
    plot_seasonal_patterns(data, y_true_dict, y_pred_dict, timestamps_dict, solar_scalers[1], load_scalers[1])
    
    print("\nForecasting complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()