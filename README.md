# Metro Interstate Traffic Analysis

A comprehensive machine learning and deep learning project for analyzing and predicting metro interstate traffic volume based on weather conditions, time features, and other factors.

## 📊 Project Overview

This project performs extensive exploratory data analysis (EDA) and implements multiple AI/ML/DL techniques to predict traffic volume on Interstate 94 in Minneapolis-St Paul. The analysis includes traditional machine learning models and advanced deep learning architectures for time series prediction.

## 🎯 Features

- **Comprehensive EDA**: Statistical analysis, visualization, and feature engineering
- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Gradient Boosting, XGBoost
- **Deep Learning Models**: Dense Neural Networks, LSTM, and GRU for time series prediction
- **Model Comparison**: Detailed performance metrics and visualizations
- **Production-Ready**: Saved models and scalers for deployment

## 📁 Dataset

The dataset (`Metro_Interstate_Traffic_Volume.csv`) contains hourly traffic volume data with:
- **Traffic Volume**: Number of vehicles (target variable)
- **Weather Features**: Temperature, rain, snow, cloud coverage
- **Temporal Features**: Date, time, holidays
- **Weather Conditions**: Main weather category and detailed descriptions
- **Size**: ~48,000 records

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Run complete analysis pipeline
python traffic_analysis.py
```

## 📈 Analysis Pipeline

1. **Data Loading & Exploration**
   - Load dataset and display basic statistics
   - Check for missing values and duplicates
   - Analyze data distributions

2. **Feature Engineering**
   - Parse datetime into temporal components
   - Extract: year, month, day, hour, day_of_week, is_weekend
   - Encode categorical variables

3. **Exploratory Data Analysis**
   - Traffic patterns by time (hourly, daily, weekly)
   - Weather impact analysis
   - Correlation analysis
   - Statistical summaries

4. **Data Preparation**
   - Train-test split (80-20)
   - Feature scaling using StandardScaler
   - Label encoding for categorical features

5. **Machine Learning Models**
   - Linear Regression (baseline)
   - Ridge & Lasso Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor

6. **Deep Learning Models**
   - Dense Neural Network (DNN)
   - LSTM (Long Short-Term Memory)
   - Optimized with dropout and callbacks

7. **Model Evaluation & Comparison**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score
   - Visual comparison charts

## 📊 Results

The analysis generates comprehensive visualizations and model comparisons:

### Visualizations
- Traffic volume distribution
- Traffic patterns by hour, day, and weather
- Correlation heatmap
- Temperature vs traffic scatter plot
- Model performance comparison

### Model Artifacts
- Trained ML models (`.pkl` files)
- Deep learning models (`.h5` files)
- Feature scaler for production use
- Complete results in JSON format

## 🔧 Technologies Used

- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: TensorFlow, Keras
- **Model Persistence**: Joblib

## 📦 Project Structure

```
Metro-interstate-traffic-analysis/
│
├── Metro_Interstate_Traffic_Volume.csv    # Dataset
├── traffic_analysis.py                    # Main analysis script
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation
├── .gitignore                            # Git ignore rules
│
├── visualizations/                        # Generated plots
│   ├── traffic_analysis_overview.png
│   ├── correlation_heatmap.png
│   ├── temp_vs_traffic.png
│   └── model_comparison.png
│
├── models/                                # Saved models
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── dnn_model.h5
│   └── lstm_model.h5
│
├── scaler.pkl                            # Feature scaler
└── model_results.json                    # Performance metrics
```

## 📊 Key Insights

- Traffic volume shows strong temporal patterns (rush hours, weekdays vs weekends)
- Weather conditions significantly impact traffic flow
- Time-based features are the strongest predictors
- Ensemble methods (Random Forest, XGBoost) perform best
- Deep learning models effectively capture complex patterns

## 🎯 Model Performance

All models are evaluated using:
- **RMSE**: Measures prediction error magnitude
- **MAE**: Average absolute prediction error
- **R² Score**: Proportion of variance explained

The best performing model is automatically identified and highlighted in the analysis output.

## 🔮 Future Enhancements

- Real-time traffic prediction API
- Integration with live weather data
- Advanced feature engineering (lag features, rolling statistics)
- Hyperparameter tuning with Bayesian optimization
- Deploy as web application with interactive dashboard

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**dude**

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Acknowledgments

- Dataset source: UCI Machine Learning Repository / Kaggle
- Traffic data from Minnesota Department of Transportation
- Weather data from OpenWeatherMap