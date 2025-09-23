import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from campus.campus_data_preprocessor import CampusEnergyPreprocessor
from campus.exploration_files.ensemble_model import EnergyEnsemble

def main():
    # Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    DATASET_PATH = os.path.join(project_root, "energy_dataset")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        return
    
    WINDOW_SIZE = 10  # Reduced window size
    TARGET_COLUMN = 'energy_consumption'
    SAMPLE_SIZE = 50000  # Manageable sample size
    
    print("="*60)
    print("CAMPUS IoT ENERGY MANAGEMENT - ENSEMBLE ANALYSIS")
    print("="*60)
    
    try:
        # Initialize preprocessor
        preprocessor = CampusEnergyPreprocessor(DATASET_PATH)
        
        # Prepare complete dataset
        print("\n1. DATA PREPARATION")
        print("-" * 30)
        data = preprocessor.prepare_complete_dataset(TARGET_COLUMN, sample_size=SAMPLE_SIZE)
        
        # Display dataset statistics
        print(f"\nDataset Overview:")
        print(f"- Total samples: {len(data):,}")
        print(f"- Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"- Buildings: {list(data['building_type'].unique())}")
        print(f"- Energy consumption range: {data['energy_consumption'].min():.2f} to {data['energy_consumption'].max():.2f}")
        
        # Building-wise statistics
        print(f"\nBuilding-wise Statistics:")
        building_stats = data.groupby('building_type')['energy_consumption'].agg(['count', 'mean', 'std']).round(2)
        print(building_stats)
        
        # Get feature subsets
        feature_subsets = preprocessor.create_feature_subsets()
        all_feature_cols = feature_subsets['all_features']
        
        print(f"\nFeature Groups:")
        for group, features in feature_subsets.items():
            print(f"- {group}: {len(features)} features")
        
        # Create sequences
        print(f"\n2. SEQUENCE CREATION")
        print("-" * 30)
        X, y = preprocessor.create_sequences(data, all_feature_cols, TARGET_COLUMN, WINDOW_SIZE)
        
        if len(X) == 0:
            print("Error: No sequences created!")
            return
            
        print(f"Sequence data shape: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        if len(X_train) == 0:
            print("Error: No training data available!")
            return
        
        # Initialize and train ensemble
        print(f"\n3. MODEL TRAINING")
        print("-" * 30)
        input_shape = (X_train.shape[1], X_train.shape[2])
        ensemble = EnergyEnsemble(input_shape)
        
        print("Training ensemble model...")
        ensemble.fit(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)  # Reduced epochs for testing
        
        # Evaluate models
        print(f"\n4. MODEL EVALUATION")
        print("-" * 30)
        results = ensemble.compare_models(X_test, y_test)
        
        # Print detailed results
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  R² Score: {metrics['R2']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  MAE: {metrics['MAE']:.2f}")
        
        # **DETAILED INDIVIDUAL MODEL ANALYSIS - ADDED HERE**
        print("\n" + "="*60)
        print("DETAILED INDIVIDUAL MODEL ANALYSIS")
        print("="*60)

        # Test all models individually
        print("\nAll Model Performance on Test Set:")
        print("-" * 40)

        # LSTM
        lstm_pred = ensemble.lstm_model.predict(X_test).flatten()
        lstm_r2 = r2_score(y_test, lstm_pred)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
        lstm_mae = mean_absolute_error(y_test, lstm_pred)
        print(f"LSTM: R² = {lstm_r2:.4f}, RMSE = {lstm_rmse:.2f}, MAE = {lstm_mae:.2f}")

        # All simple models
        print("\nSimple Models Performance:")
        simple_model_results = {}
        for model_name in ensemble.simple_models.models.keys():
            try:
                pred = ensemble.simple_models.predict_single_model(model_name, X_test)
                r2 = r2_score(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                simple_model_results[model_name] = {'r2': r2, 'rmse': rmse, 'mae': mae}
                print(f"{model_name}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
            except Exception as e:
                print(f"{model_name}: Error - {e}")

        # Ensemble performance breakdown
        ensemble_pred = ensemble.predict(X_test)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        print(f"\nFinal Ensemble: R² = {ensemble_r2:.4f}, RMSE = {ensemble_rmse:.2f}, MAE = {ensemble_mae:.2f}")

        print("\n" + "="*40)
        print("ENSEMBLE WEIGHT ANALYSIS")
        print("="*40)
        total_weight = sum(ensemble.weights.values())
        print(f"Total weight sum: {total_weight:.4f}")
        
        for model, weight in ensemble.weights.items():
            percentage = weight * 100
            print(f"{model}: Weight = {weight:.4f} ({percentage:.1f}%)")
            
            # Show what this weight means in terms of performance contribution
            if model == 'lstm':
                expected_contrib = weight * lstm_r2
                print(f"  → Expected contribution to ensemble R²: {expected_contrib:.4f}")
            elif model in simple_model_results:
                expected_contrib = weight * simple_model_results[model]['r2']
                print(f"  → Expected contribution to ensemble R²: {expected_contrib:.4f}")

        # Performance analysis
        print("\n" + "="*40)
        print("PERFORMANCE ANALYSIS")
        print("="*40)
        
        # Find best individual model
        all_r2_scores = {'lstm': lstm_r2}
        all_r2_scores.update({k: v['r2'] for k, v in simple_model_results.items()})
        
        best_model = max(all_r2_scores.items(), key=lambda x: x[1])
        worst_model = min(all_r2_scores.items(), key=lambda x: x[1])
        
        print(f"Best individual model: {best_model[0]} (R² = {best_model[1]:.4f})")
        print(f"Worst individual model: {worst_model[0]} (R² = {worst_model[1]:.4f})")
        print(f"Ensemble performance: R² = {ensemble_r2:.4f}")
        
        if ensemble_r2 > best_model[1]:
            improvement = ((ensemble_r2 - best_model[1]) / best_model[1]) * 100
            print(f"✅ Ensemble IMPROVED by {improvement:.2f}% over best individual model")
        else:
            degradation = ((best_model[1] - ensemble_r2) / best_model[1]) * 100
            print(f"❌ Ensemble UNDERPERFORMED by {degradation:.2f}% vs best individual model")
            print(f"   Recommendation: Use {best_model[0]} alone or adjust ensemble weights")

        # Model contribution analysis
        print(f"\nModel Contribution Analysis:")
        weighted_sum = sum([ensemble.weights.get('lstm', 0) * lstm_r2] + 
                          [ensemble.weights.get(k, 0) * v['r2'] for k, v in simple_model_results.items()])
        print(f"Theoretical ensemble R² (weighted sum): {weighted_sum:.4f}")
        print(f"Actual ensemble R²: {ensemble_r2:.4f}")
        print(f"Ensemble synergy effect: {(ensemble_r2 - weighted_sum):.4f}")

        print(f"\nEnsemble weights: {ensemble.weights}")
        
        # Save results summary
        save_results_summary(results, ensemble, lstm_r2, simple_model_results, ensemble_r2)
        
        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def save_results_summary(results, ensemble, lstm_r2, simple_model_results, ensemble_r2):
    """Save analysis results to file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "analysis_summary.txt"), "w") as f:
        f.write("CAMPUS ENERGY ENSEMBLE ANALYSIS RESULTS\n")
        f.write("="*50 + "\n\n")
        
        f.write("INDIVIDUAL MODEL PERFORMANCE:\n")
        f.write(f"LSTM: R² = {lstm_r2:.4f}\n")
        for model, metrics in simple_model_results.items():
            f.write(f"{model}: R² = {metrics['r2']:.4f}\n")
        f.write(f"Ensemble: R² = {ensemble_r2:.4f}\n\n")
        
        f.write("ENSEMBLE WEIGHTS:\n")
        for model, weight in ensemble.weights.items():
            f.write(f"  {model}: {weight:.4f} ({weight*100:.1f}%)\n")
        
        f.write(f"\nFINAL COMPARISON:\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  R² Score: {metrics['R2']:.4f}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
            f.write(f"  MAE: {metrics['MAE']:.2f}\n\n")
    
    print(f"Results saved to: {os.path.join(results_dir, 'analysis_summary.txt')}")

if __name__ == "__main__":
    main()
