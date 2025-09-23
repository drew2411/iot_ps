"""
Generic implementation for any energy dataset
For campus-specific analysis, use main_campus.py
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import DataPreprocessor
from ensemble_model import EnergyEnsemble

def main():
    # Configuration
    DATA_PATH = "path_to_your_dataset.csv"  # Replace with your dataset path
    WINDOW_SIZE = 24  # 24 hours lookback
    TARGET_COLUMN = 'energy_consumption'
    
    print("Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data (you'll need to replace this with actual I-BLEND dataset)
    # For demonstration, creating sample data structure
    # data = preprocessor.load_data(DATA_PATH)
    
    # Sample data creation (replace with actual data loading)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=8760, freq='H')  # 1 year hourly data
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': np.random.normal(100, 20, 8760) + 50 * np.sin(np.arange(8760) * 2 * np.pi / 24),
        'temperature': np.random.normal(20, 10, 8760),
        'humidity': np.random.normal(60, 15, 8760),
        'solar_radiation': np.random.normal(200, 50, 8760),
        'building_type': np.random.choice(['office', 'residential', 'commercial'], 8760),
        'occupancy': np.random.uniform(0, 1, 8760)
    })
    
    # Preprocess data
    data = preprocessor.preprocess_data(sample_data, TARGET_COLUMN)
    
    # Get feature subsets
    feature_subsets = preprocessor.create_feature_subsets(data)
    all_feature_cols = feature_subsets['all_features']
    
    print(f"Features used: {all_feature_cols}")
    print(f"Data shape after preprocessing: {data.shape}")
    
    # Create sequences for LSTM
    X, y = preprocessor.create_sequences(
        data, all_feature_cols, TARGET_COLUMN, WINDOW_SIZE
    )
    
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Initialize and train ensemble
    input_shape = (X_train.shape[1], X_train.shape[2])
    ensemble = EnergyEnsemble(input_shape)
    
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = ensemble.compare_models(X_test, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MSE: {metrics['MSE']:.4f}")
    
    # Visualize results
    ensemble_pred = ensemble.predict(X_test)
    lstm_pred = ensemble.lstm_model.predict(X_test).flatten()
    
    # Inverse transform predictions for plotting
    y_test_actual = preprocessor.scalers['target'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    ensemble_pred_actual = preprocessor.scalers['target'].inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
    lstm_pred_actual = preprocessor.scalers['target'].inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Predictions vs Actual
    plt.subplot(2, 2, 1)
    plt.plot(y_test_actual[:200], label='Actual', alpha=0.7)
    plt.plot(ensemble_pred_actual[:200], label='Ensemble', alpha=0.7)
    plt.plot(lstm_pred_actual[:200], label='LSTM Only', alpha=0.7)
    plt.legend()
    plt.title('Energy Consumption Predictions (First 200 samples)')
    plt.ylabel('Energy Consumption')
    
    # Plot 2: Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_test_actual, ensemble_pred_actual, alpha=0.5, label='Ensemble')
    plt.scatter(y_test_actual, lstm_pred_actual, alpha=0.5, label='LSTM Only')
    plt.plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Actual vs Predicted')
    
    # Plot 3: Residuals
    plt.subplot(2, 2, 3)
    ensemble_residuals = y_test_actual - ensemble_pred_actual
    lstm_residuals = y_test_actual - lstm_pred_actual
    plt.hist(ensemble_residuals, alpha=0.5, label='Ensemble Residuals', bins=30)
    plt.hist(lstm_residuals, alpha=0.5, label='LSTM Residuals', bins=30)
    plt.legend()
    plt.title('Residuals Distribution')
    plt.xlabel('Residual Value')
    
    # Plot 4: Model weights
    plt.subplot(2, 2, 4)
    weights = list(ensemble.weights.values())
    labels = list(ensemble.weights.keys())
    plt.bar(labels, weights)
    plt.title('Ensemble Model Weights')
    plt.xticks(rotation=45)
    plt.ylabel('Weight')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nEnsemble weights: {ensemble.weights}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
