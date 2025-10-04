"""
Traffic Volume Prediction Script
Use trained models to predict traffic volume for new data
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import argparse

def predict_traffic(temp, rain, snow, clouds, weather_main, holiday, 
                   date_time, model_path='xgboost_model.pkl'):
    """
    Predict traffic volume using a trained model
    
    Parameters:
    -----------
    temp : float
        Temperature in Kelvin
    rain : float
        Rain volume in mm (past 1 hour)
    snow : float
        Snow volume in mm (past 1 hour)
    clouds : int
        Cloud coverage percentage (0-100)
    weather_main : str
        Main weather category (e.g., 'Clear', 'Clouds', 'Rain')
    holiday : str
        Holiday name or 'None'
    date_time : str
        Date and time in format 'DD-MM-YYYY HH:MM'
    model_path : str
        Path to the trained model file
    
    Returns:
    --------
    float : Predicted traffic volume
    """
    
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load('scaler.pkl')
    
    # Parse datetime
    dt = datetime.strptime(date_time, '%d-%m-%Y %H:%M')
    
    # Extract time features
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    day_of_week = dt.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Encode categorical variables (simplified - use same encoding as training)
    weather_mapping = {
        'Clear': 0, 'Clouds': 1, 'Mist': 2, 'Rain': 3, 'Snow': 4,
        'Drizzle': 5, 'Haze': 6, 'Thunderstorm': 7, 'Fog': 8, 'Smoke': 9, 'Squall': 10
    }
    weather_encoded = weather_mapping.get(weather_main, 0)
    holiday_encoded = 0 if holiday == 'None' else 1
    
    # Create feature vector
    features = np.array([[temp, rain, snow, clouds, weather_encoded, holiday_encoded,
                         year, month, day, hour, day_of_week, is_weekend]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction

def main():
    """Command-line interface for traffic prediction"""
    parser = argparse.ArgumentParser(description='Predict metro interstate traffic volume')
    parser.add_argument('--temp', type=float, required=True, help='Temperature in Kelvin')
    parser.add_argument('--rain', type=float, default=0.0, help='Rain volume (mm)')
    parser.add_argument('--snow', type=float, default=0.0, help='Snow volume (mm)')
    parser.add_argument('--clouds', type=int, default=0, help='Cloud coverage (0-100)')
    parser.add_argument('--weather', type=str, default='Clear', help='Weather condition')
    parser.add_argument('--holiday', type=str, default='None', help='Holiday name')
    parser.add_argument('--datetime', type=str, required=True, help='Date-time (DD-MM-YYYY HH:MM)')
    parser.add_argument('--model', type=str, default='xgboost_model.pkl', help='Model file path')
    
    args = parser.parse_args()
    
    # Make prediction
    try:
        prediction = predict_traffic(
            temp=args.temp,
            rain=args.rain,
            snow=args.snow,
            clouds=args.clouds,
            weather_main=args.weather,
            holiday=args.holiday,
            date_time=args.datetime,
            model_path=args.model
        )
        
        print("\n" + "="*60)
        print("TRAFFIC VOLUME PREDICTION")
        print("="*60)
        print(f"\nInput Parameters:")
        print(f"  Date/Time: {args.datetime}")
        print(f"  Temperature: {args.temp} K ({args.temp - 273.15:.1f} °C)")
        print(f"  Weather: {args.weather}")
        print(f"  Rain: {args.rain} mm")
        print(f"  Snow: {args.snow} mm")
        print(f"  Cloud Coverage: {args.clouds}%")
        print(f"  Holiday: {args.holiday}")
        print(f"\n{'='*60}")
        print(f"Predicted Traffic Volume: {int(prediction)} vehicles/hour")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage
    print("\nExample Usage:")
    print("-" * 60)
    print("python predict_traffic.py \\")
    print("  --temp 285.5 \\")
    print("  --rain 0.0 \\")
    print("  --snow 0.0 \\")
    print("  --clouds 20 \\")
    print("  --weather Clear \\")
    print("  --holiday None \\")
    print("  --datetime '15-06-2024 17:00' \\")
    print("  --model xgboost_model.pkl")
    print("-" * 60 + "\n")
    
    main()
