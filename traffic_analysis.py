"""
Metro Interstate Traffic Volume Analysis
Comprehensive EDA and ML/DL modeling for traffic prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import json

class MetroTrafficAnalysis:
    """Complete traffic analysis pipeline with EDA, ML and DL models"""
    
    def __init__(self, data_path):
        """Initialize the analysis pipeline"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nDataset Info:")
        print(self.df.info())
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        return self.df
    
    def perform_eda(self):
        """Perform comprehensive Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Check for missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Parse datetime
        self.df['date_time'] = pd.to_datetime(self.df['date_time'], format='%d-%m-%Y %H:%M')
        
        # Extract time-based features
        self.df['year'] = self.df['date_time'].dt.year
        self.df['month'] = self.df['date_time'].dt.month
        self.df['day'] = self.df['date_time'].dt.day
        self.df['hour'] = self.df['date_time'].dt.hour
        self.df['day_of_week'] = self.df['date_time'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        print("\nFeature Engineering Completed:")
        print(f"- Extracted: year, month, day, hour, day_of_week, is_weekend")
        
        # Traffic volume statistics
        print(f"\nTraffic Volume Statistics:")
        print(f"Mean: {self.df['traffic_volume'].mean():.2f}")
        print(f"Median: {self.df['traffic_volume'].median():.2f}")
        print(f"Std Dev: {self.df['traffic_volume'].std():.2f}")
        print(f"Min: {self.df['traffic_volume'].min()}")
        print(f"Max: {self.df['traffic_volume'].max()}")
        
        # Weather statistics
        print(f"\nWeather Statistics:")
        print(f"Temperature (K) - Mean: {self.df['temp'].mean():.2f}, Range: [{self.df['temp'].min():.2f}, {self.df['temp'].max():.2f}]")
        print(f"Rain (mm) - Mean: {self.df['rain_1h'].mean():.4f}, Max: {self.df['rain_1h'].max():.2f}")
        print(f"Snow (mm) - Mean: {self.df['snow_1h'].mean():.4f}, Max: {self.df['snow_1h'].max():.2f}")
        print(f"Cloud Coverage (%) - Mean: {self.df['clouds_all'].mean():.2f}")
        
        # Categorical variables
        print(f"\nCategorical Variables:")
        print(f"Weather Main Categories: {self.df['weather_main'].nunique()}")
        print(self.df['weather_main'].value_counts())
        print(f"\nHoliday Values: {self.df['holiday'].nunique()}")
        print(self.df['holiday'].value_counts().head())
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        # Create figure directory
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Traffic Volume Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].hist(self.df['traffic_volume'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Traffic Volume Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Traffic Volume')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Traffic by Hour
        hourly_traffic = self.df.groupby('hour')['traffic_volume'].mean()
        axes[0, 1].plot(hourly_traffic.index, hourly_traffic.values, marker='o', linewidth=2)
        axes[0, 1].set_title('Average Traffic by Hour of Day', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Average Traffic Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Traffic by Day of Week
        daily_traffic = self.df.groupby('day_of_week')['traffic_volume'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(range(7), daily_traffic.values, color='skyblue', edgecolor='black')
        axes[1, 0].set_xticks(range(7))
        axes[1, 0].set_xticklabels(days)
        axes[1, 0].set_title('Average Traffic by Day of Week', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Average Traffic Volume')
        
        # 4. Traffic by Weather
        weather_traffic = self.df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False)
        axes[1, 1].barh(range(len(weather_traffic)), weather_traffic.values, color='coral', edgecolor='black')
        axes[1, 1].set_yticks(range(len(weather_traffic)))
        axes[1, 1].set_yticklabels(weather_traffic.index)
        axes[1, 1].set_title('Average Traffic by Weather Condition', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Average Traffic Volume')
        
        plt.tight_layout()
        plt.savefig('visualizations/traffic_analysis_overview.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/traffic_analysis_overview.png")
        plt.close()
        
        # 5. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        numeric_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
                        'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend']
        correlation = self.df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/correlation_heatmap.png")
        plt.close()
        
        # 6. Temperature vs Traffic
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(self.df['temp'], self.df['traffic_volume'], alpha=0.1, s=1)
        ax.set_title('Temperature vs Traffic Volume', fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Traffic Volume')
        plt.tight_layout()
        plt.savefig('visualizations/temp_vs_traffic.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/temp_vs_traffic.png")
        plt.close()
        
        print("\nAll visualizations created successfully!")
    
    def prepare_data(self):
        """Prepare data for machine learning models"""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Encode categorical variables
        le_weather = LabelEncoder()
        le_holiday = LabelEncoder()
        
        self.df['weather_main_encoded'] = le_weather.fit_transform(self.df['weather_main'])
        self.df['holiday_encoded'] = le_holiday.fit_transform(self.df['holiday'])
        
        # Select features for modeling
        feature_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 
                       'weather_main_encoded', 'holiday_encoded',
                       'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend']
        
        X = self.df[feature_cols]
        y = self.df['traffic_volume']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Number of features: {self.X_train.shape[1]}")
        print(f"Features: {feature_cols}")
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        print("\n✓ Scaler saved to scaler.pkl")
        
    def train_ml_models(self):
        """Train traditional machine learning models"""
        print("\n" + "="*80)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*80)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            self.results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            print(f"  Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            print(f"  Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
            print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            
            # Save best models
            if name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
                print(f"  ✓ Model saved")
    
    def train_dl_models(self):
        """Train deep learning models"""
        print("\n" + "="*80)
        print("TRAINING DEEP LEARNING MODELS")
        print("="*80)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # 1. Simple Dense Neural Network
        print("\nTraining Dense Neural Network...")
        dnn_model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        dnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history_dnn = dnn_model.fit(
            self.X_train_scaled, self.y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate DNN
        y_pred_train_dnn = dnn_model.predict(self.X_train_scaled, verbose=0).flatten()
        y_pred_test_dnn = dnn_model.predict(self.X_test_scaled, verbose=0).flatten()
        
        self.results['Deep Neural Network'] = {
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train_dnn)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test_dnn)),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train_dnn),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test_dnn),
            'train_r2': r2_score(self.y_train, y_pred_train_dnn),
            'test_r2': r2_score(self.y_test, y_pred_test_dnn)
        }
        
        print(f"  Train RMSE: {self.results['Deep Neural Network']['train_rmse']:.2f}")
        print(f"  Test RMSE: {self.results['Deep Neural Network']['test_rmse']:.2f}")
        print(f"  Test R²: {self.results['Deep Neural Network']['test_r2']:.4f}")
        
        dnn_model.save('dnn_model.h5')
        print("  ✓ Model saved to dnn_model.h5")
        
        # 2. LSTM for Time Series
        print("\nTraining LSTM Network...")
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = self.X_train_scaled.reshape((self.X_train_scaled.shape[0], 1, self.X_train_scaled.shape[1]))
        X_test_lstm = self.X_test_scaled.reshape((self.X_test_scaled.shape[0], 1, self.X_test_scaled.shape[1]))
        
        lstm_model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(1, self.X_train_scaled.shape[1])),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history_lstm = lstm_model.fit(
            X_train_lstm, self.y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate LSTM
        y_pred_train_lstm = lstm_model.predict(X_train_lstm, verbose=0).flatten()
        y_pred_test_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        
        self.results['LSTM'] = {
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred_train_lstm)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test_lstm)),
            'train_mae': mean_absolute_error(self.y_train, y_pred_train_lstm),
            'test_mae': mean_absolute_error(self.y_test, y_pred_test_lstm),
            'train_r2': r2_score(self.y_train, y_pred_train_lstm),
            'test_r2': r2_score(self.y_test, y_pred_test_lstm)
        }
        
        print(f"  Train RMSE: {self.results['LSTM']['train_rmse']:.2f}")
        print(f"  Test RMSE: {self.results['LSTM']['test_rmse']:.2f}")
        print(f"  Test R²: {self.results['LSTM']['test_r2']:.4f}")
        
        lstm_model.save('lstm_model.h5')
        print("  ✓ Model saved to lstm_model.h5")
        
    def compare_models(self):
        """Compare all models and visualize results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(2)
        
        print("\nModel Performance Summary:")
        print(comparison_df.to_string())
        
        # Save results to JSON
        with open('model_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        print("\n✓ Results saved to model_results.json")
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        test_rmse = [self.results[m]['test_rmse'] for m in models]
        test_mae = [self.results[m]['test_mae'] for m in models]
        test_r2 = [self.results[m]['test_r2'] for m in models]
        
        # RMSE comparison
        axes[0, 0].barh(models, test_rmse, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('RMSE')
        axes[0, 0].set_title('Test RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # MAE comparison
        axes[0, 1].barh(models, test_mae, color='coral', edgecolor='black')
        axes[0, 1].set_xlabel('MAE')
        axes[0, 1].set_title('Test MAE Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].invert_yaxis()
        
        # R² comparison
        axes[1, 0].barh(models, test_r2, color='seagreen', edgecolor='black')
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_title('Test R² Score Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Best model highlight
        best_model = min(self.results.items(), key=lambda x: x[1]['test_rmse'])
        axes[1, 1].text(0.5, 0.6, 'Best Model', ha='center', va='center', 
                       fontsize=20, fontweight='bold')
        axes[1, 1].text(0.5, 0.4, best_model[0], ha='center', va='center', 
                       fontsize=18, color='darkblue')
        axes[1, 1].text(0.5, 0.25, f'Test RMSE: {best_model[1]["test_rmse"]:.2f}', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].text(0.5, 0.15, f'Test R²: {best_model[1]["test_r2"]:.4f}', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/model_comparison.png")
        plt.close()
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model[0]}")
        print(f"Test RMSE: {best_model[1]['test_rmse']:.2f}")
        print(f"Test MAE: {best_model[1]['test_mae']:.2f}")
        print(f"Test R²: {best_model[1]['test_r2']:.4f}")
        print(f"{'='*80}")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*80)
        print("METRO INTERSTATE TRAFFIC ANALYSIS")
        print("Complete ML/DL Pipeline")
        print("="*80)
        
        self.load_data()
        self.perform_eda()
        self.create_visualizations()
        self.prepare_data()
        self.train_ml_models()
        self.train_dl_models()
        self.compare_models()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  - visualizations/traffic_analysis_overview.png")
        print("  - visualizations/correlation_heatmap.png")
        print("  - visualizations/temp_vs_traffic.png")
        print("  - visualizations/model_comparison.png")
        print("  - scaler.pkl")
        print("  - random_forest_model.pkl")
        print("  - xgboost_model.pkl")
        print("  - gradient_boosting_model.pkl")
        print("  - dnn_model.h5")
        print("  - lstm_model.h5")
        print("  - model_results.json")
        print("\nAll models trained and evaluated successfully!")

def main():
    """Main execution function"""
    # Initialize analysis
    analysis = MetroTrafficAnalysis('Metro_Interstate_Traffic_Volume.csv')
    
    # Run complete pipeline
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main()
