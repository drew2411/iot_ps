import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimpleModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_subsets = {}
        
    def prepare_features_for_simple_models(self, X_sequences, feature_subset_names):
        """Convert LSTM sequences to flat features for simple models"""
        # Take the last timestep from sequences (most recent features)
        X_flat = X_sequences[:, -1, :]  # Shape: (samples, features)
        
        # Create different feature subsets
        # Note: This assumes you know the feature order from preprocessing
        feature_maps = {
            'temporal': slice(0, 9),      # First 9 features are temporal
            'weather': slice(9, 12),      # Next 3 are weather
            'energy_history': slice(12, 16),  # Next 4 are energy history
            'building': slice(16, 18),    # Last 2 are building features
            'all_features': slice(None)   # All features
        }
        
        feature_data = {}
        for subset_name in feature_subset_names:
            if subset_name in feature_maps:
                feature_data[subset_name] = X_flat[:, feature_maps[subset_name]]
        
        return feature_data
    
    def train_models(self, X_train_sequences, y_train, X_val_sequences, y_val):
        """Train simple models on different feature subsets"""
        feature_subset_names = ['temporal', 'weather', 'energy_history', 'building', 'all_features']
        
        # Prepare flat features
        train_features = self.prepare_features_for_simple_models(X_train_sequences, feature_subset_names)
        val_features = self.prepare_features_for_simple_models(X_val_sequences, feature_subset_names)
        
        # Train models for each subset
        for subset_name, X_subset in train_features.items():
            # Scale features for this subset
            scaler = StandardScaler()
            X_subset_scaled = scaler.fit_transform(X_subset)
            X_val_subset_scaled = scaler.transform(val_features[subset_name])
            
            self.scalers[subset_name] = scaler
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_subset_scaled, y_train)
            self.models[f'lr_{subset_name}'] = lr_model
            
            # Ridge Regression (regularized)
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_subset_scaled, y_train)
            self.models[f'ridge_{subset_name}'] = ridge_model
            
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_subset_scaled, y_train)
            self.models[f'rf_{subset_name}'] = rf_model
            
        # Evaluate all models
        self.evaluate_all_models(X_val_sequences, y_val)
        
    def predict_single_model(self, model_name, X_sequences):
        """Predict using a single simple model"""
        # Extract subset name from model name
        subset_name = model_name.split('_', 1)[1]
        
        # Prepare features
        feature_data = self.prepare_features_for_simple_models(
            X_sequences, [subset_name]
        )
        X_subset = feature_data[subset_name]
        
        # Scale features
        X_subset_scaled = self.scalers[subset_name].transform(X_subset)
        
        # Predict
        return self.models[model_name].predict(X_subset_scaled)
    
    def evaluate_all_models(self, X_val, y_val):
        """Evaluate all simple models"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                y_pred = self.predict_single_model(model_name, X_val)
                
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                results[model_name] = {
                    'MSE': mse,
                    'RMSE': np.sqrt(mse),
                    'MAE': mae,
                    'R2': r2
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                
        return results
