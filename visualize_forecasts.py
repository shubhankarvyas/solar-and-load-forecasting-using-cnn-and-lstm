import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import argparse

def load_data(data_path):
    """Load data from CSV file."""
    return pd.read_csv(data_path, index_col='timestamp', parse_dates=True)

def create_advanced_plots(data, solar_preds, load_preds, output_dir='results'):
    """Create advanced visualization plots for solar and load forecasts.
    
    Args:
        data: DataFrame with actual data
        solar_preds: Predicted solar generation values
        load_preds: Predicted load values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Daily profile plot (average by hour)
    plt.figure(figsize=(15, 6))
    
    # Add hour as a column if not present
    if 'hour' not in data.columns:
        data['hour'] = data.index.hour
    
    # Solar generation by hour
    plt.subplot(1, 2, 1)
    hourly_solar = data.groupby('hour')['solar_generation'].mean()
    plt.plot(hourly_solar.index, hourly_solar.values, 'o-', label='Actual')
    plt.title('Average Solar Generation by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Solar Generation (kW)')
    plt.grid(True)
    plt.xticks(range(0, 24, 2))
    
    # Load by hour
    plt.subplot(1, 2, 2)
    hourly_load = data.groupby('hour')['load'].mean()
    plt.plot(hourly_load.index, hourly_load.values, 'o-', label='Actual')
    plt.title('Average Load by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (kW)')
    plt.grid(True)
    plt.xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/daily_profiles.png')
    plt.close()
    
    # 2. Seasonal patterns (monthly averages)
    plt.figure(figsize=(15, 6))
    
    # Add month as a column if not present
    if 'month' not in data.columns:
        data['month'] = data.index.month
    
    # Solar generation by month
    plt.subplot(1, 2, 1)
    monthly_solar = data.groupby('month')['solar_generation'].mean()
    plt.plot(monthly_solar.index, monthly_solar.values, 'o-', label='Actual')
    plt.title('Average Solar Generation by Month')
    plt.xlabel('Month')
    plt.ylabel('Solar Generation (kW)')
    plt.grid(True)
    plt.xticks(range(1, 13))
    
    # Load by month
    plt.subplot(1, 2, 2)
    monthly_load = data.groupby('month')['load'].mean()
    plt.plot(monthly_load.index, monthly_load.values, 'o-', label='Actual')
    plt.title('Average Load by Month')
    plt.xlabel('Month')
    plt.ylabel('Load (kW)')
    plt.grid(True)
    plt.xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/seasonal_patterns.png')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_columns = ['temperature', 'cloud_cover', 'solar_irradiance', 'solar_generation', 'load']
    corr_matrix = data[corr_columns].corr()
    
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Feature Correlation Heatmap')
    
    # Add correlation values
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                     ha='center', va='center', color='black')
    
    plt.xticks(range(len(corr_columns)), corr_columns, rotation=45)
    plt.yticks(range(len(corr_columns)), corr_columns)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    # 4. Forecast error distribution
    if solar_preds is not None and load_preds is not None:
        plt.figure(figsize=(15, 6))
        
        # Solar forecast errors
        plt.subplot(1, 2, 1)
        solar_errors = solar_preds - data['solar_generation'].values[-len(solar_preds):]
        plt.hist(solar_errors, bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Solar Forecast Error Distribution')
        plt.xlabel('Forecast Error (kW)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Load forecast errors
        plt.subplot(1, 2, 2)
        load_errors = load_preds - data['load'].values[-len(load_preds):]
        plt.hist(load_errors, bins=20, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Load Forecast Error Distribution')
        plt.xlabel('Forecast Error (kW)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_distribution.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize solar and load forecasts')
    parser.add_argument('--data', type=str, default='results/data.csv',
                        help='Path to the data CSV file')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save visualization results')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    try:
        data = load_data(args.data)
        print(f"Data loaded successfully with {len(data)} records.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check if prediction results exist
    solar_preds = None
    load_preds = None
    
    # Create visualizations
    print("Creating visualizations...")
    create_advanced_plots(data, solar_preds, load_preds, args.output)
    
    print(f"Visualizations saved to {args.output} directory.")

if __name__ == "__main__":
    main()