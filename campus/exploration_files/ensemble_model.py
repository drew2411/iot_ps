# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from campus.decode_lstm import DecodeLSTM
# from campus.simple_models import SimpleModels

# class EnergyEnsemble:
#     def __init__(self, input_shape):
#         self.lstm_model = DecodeLSTM(input_shape)
#         self.simple_models = SimpleModels()
#         self.weights = {}
#         self.validation_scores = {}
        
#     def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
#         """Train all models in the ensemble"""
#         print("Training LSTM model...")
#         self.lstm_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
#         print("Training simple models...")
#         self.simple_models.train_models(X_train, y_train, X_val, y_val)
        
#         print("Calculating optimal ensemble weights...")
#         self.calculate_weights_improved(X_val, y_val)
        
#     def calculate_weights_improved(self, X_val, y_val):
#         """Improved ensemble weight calculation - only use good models"""
#         # Get predictions from all models
#         lstm_pred = self.lstm_model.predict(X_val).flatten()
        
#         # Evaluate LSTM performance
#         lstm_r2 = r2_score(y_val, lstm_pred)
#         print(f"LSTM R² on validation: {lstm_r2:.4f}")
        
#         # Get simple model predictions and scores
#         model_scores = {}
#         model_scores['lstm'] = lstm_r2
        
#         for model_name in self.simple_models.models.keys():
#             try:
#                 pred = self.simple_models.predict_single_model(model_name, X_val)
#                 r2 = r2_score(y_val, pred)
#                 model_scores[model_name] = r2
#                 print(f"{model_name} R² on validation: {r2:.4f}")
#             except:
#                 continue
        
#         # Only include models that perform reasonably well
#         min_r2_threshold = 0.1  # Minimum acceptable R²
#         good_models = {k: v for k, v in model_scores.items() if v >= min_r2_threshold}
        
#         print(f"Models meeting threshold (R² >= {min_r2_threshold}): {list(good_models.keys())}")
        
#         if len(good_models) == 0:
#             # Fallback: use only LSTM if no models meet threshold
#             self.weights = {'lstm': 1.0}
#             print("No models met threshold. Using LSTM only.")
#         elif lstm_r2 > max([v for k, v in good_models.items() if k != 'lstm'], default=0):
#             # If LSTM is clearly the best, give it higher weight
#             total_score = sum(good_models.values())
#             self.weights = {}
            
#             for model, score in good_models.items():
#                 if model == 'lstm':
#                     # Give LSTM 60% weight if it's the best
#                     self.weights[model] = 0.6 + (0.4 * score / total_score)
#                 else:
#                     # Distribute remaining weight among other good models
#                     self.weights[model] = 0.4 * score / (total_score - good_models['lstm'])
            
#             # Normalize weights to sum to 1
#             weight_sum = sum(self.weights.values())
#             self.weights = {k: v/weight_sum for k, v in self.weights.items()}
#         else:
#             # Standard performance-based weighting
#             total_score = sum(good_models.values())
#             self.weights = {model: score/total_score for model, score in good_models.items()}
        
#         # Store validation scores
#         self.validation_scores = model_scores
        
#         print("Final ensemble weights:", {k: f"{v:.4f}" for k, v in self.weights.items()})
        
#     def predict(self, X_test):
#         """Make ensemble predictions using only weighted good models"""
#         predictions = {}
        
#         # LSTM prediction
#         if 'lstm' in self.weights:
#             lstm_pred = self.lstm_model.predict(X_test).flatten()
#             predictions['lstm'] = lstm_pred
        
#         # Simple model predictions
#         for model_name in self.simple_models.models.keys():
#             if model_name in self.weights:
#                 try:
#                     pred = self.simple_models.predict_single_model(model_name, X_test)
#                     predictions[model_name] = pred
#                 except:
#                     continue
        
#         # Weighted ensemble
#         if len(predictions) == 0:
#             # Fallback to LSTM if no predictions available
#             return self.lstm_model.predict(X_test).flatten()
        
#         # Calculate ensemble prediction
#         ensemble_pred = np.zeros_like(list(predictions.values())[0])
#         for model_name, weight in self.weights.items():
#             if model_name in predictions:
#                 ensemble_pred += weight * predictions[model_name]
        
#         return ensemble_pred
    
#     def evaluate(self, X_test, y_test):
#         """Evaluate ensemble performance"""
#         y_pred = self.predict(X_test)
        
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mse)
        
#         metrics = {
#             'MSE': mse,
#             'RMSE': rmse,
#             'MAE': mae,
#             'R2': r2
#         }
        
#         return metrics
    
#     def compare_models(self, X_test, y_test):
#         """Compare ensemble with individual models"""
#         results = {}
        
#         # Ensemble performance
#         results['ensemble'] = self.evaluate(X_test, y_test)
        
#         # LSTM performance
#         lstm_pred = self.lstm_model.predict(X_test).flatten()
#         results['lstm_only'] = {
#             'MSE': mean_squared_error(y_test, lstm_pred),
#             'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
#             'MAE': mean_absolute_error(y_test, lstm_pred),
#             'R2': r2_score(y_test, lstm_pred)
#         }
        
#         # Best performing simple models
#         best_models = []
#         for model_name in self.simple_models.models.keys():
#             try:
#                 pred = self.simple_models.predict_single_model(model_name, X_test)
#                 r2 = r2_score(y_test, pred)
#                 if r2 > 0.2:  # Only include reasonably good models
#                     results[f'{model_name}'] = {
#                         'MSE': mean_squared_error(y_test, pred),
#                         'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
#                         'MAE': mean_absolute_error(y_test, pred),
#                         'R2': r2
#                     }
#                     best_models.append((model_name, r2))
#             except:
#                 continue
        
#         # Sort and get best simple model
#         best_models.sort(key=lambda x: x[1], reverse=True)
#         if best_models:
#             best_simple_name, best_simple_r2 = best_models[0]
#             print(f"Best simple model: {best_simple_name} (R² = {best_simple_r2:.4f})")
                
#         return results


''' Trial 2'''

# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from campus.decode_lstm import DecodeLSTM
# from campus.simple_models import SimpleModels

# class EnergyEnsemble:
#     def __init__(self, input_shape):
#         self.lstm_model = DecodeLSTM(input_shape)
#         self.simple_models = SimpleModels()
#         self.weights = {}
#         self.validation_scores = {}
        
#     def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
#         """Train all models in the ensemble"""
#         print("Training LSTM model...")
#         self.lstm_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
#         print("Training simple models...")
#         self.simple_models.train_models(X_train, y_train, X_val, y_val)
        
#         print("Calculating CORRECTED ensemble weights...")
#         self.calculate_weights_corrected(X_val, y_val)
        
#     def calculate_weights_corrected(self, X_val, y_val):
#         """CORRECTED ensemble weight calculation"""
#         # Get predictions from all models
#         lstm_pred = self.lstm_model.predict(X_val).flatten()
        
#         # Evaluate LSTM performance
#         lstm_r2 = r2_score(y_val, lstm_pred)
#         print(f"LSTM R² on validation: {lstm_r2:.4f}")
        
#         # Get simple model predictions and scores
#         model_scores = {}
#         model_scores['lstm'] = lstm_r2
        
#         for model_name in self.simple_models.models.keys():
#             try:
#                 pred = self.simple_models.predict_single_model(model_name, X_val)
#                 r2 = r2_score(y_val, pred)
#                 model_scores[model_name] = r2
#                 print(f"{model_name} R² on validation: {r2:.4f}")
#             except:
#                 continue
        
#         # **CORRECTED LOGIC**: Only use models with POSITIVE R² > threshold
#         min_r2_threshold = 0.1  # Minimum acceptable R²
#         good_models = {k: v for k, v in model_scores.items() if v >= min_r2_threshold}
        
#         print(f"Models meeting threshold (R² >= {min_r2_threshold}): {list(good_models.keys())}")
        
#         if len(good_models) == 0:
#             # No models meet threshold - use best available model
#             best_model = max(model_scores.items(), key=lambda x: x[1])
#             self.weights = {best_model[0]: 1.0}
#             print(f"No models met threshold. Using best model: {best_model[0]} (R² = {best_model[1]:.4f})")
            
#         elif len(good_models) == 1:
#             # Only one good model - use it exclusively
#             model_name = list(good_models.keys())[0]
#             self.weights = {model_name: 1.0}
#             print(f"Only one good model. Using: {model_name} (R² = {good_models[model_name]:.4f})")
            
#         else:
#             # Multiple good models - weight by performance
#             # Use squared R² to emphasize better performers
#             performance_weights = {k: max(v, 0.01)**2 for k, v in good_models.items()}
#             total_weight = sum(performance_weights.values())
            
#             self.weights = {k: w/total_weight for k, w in performance_weights.items()}
            
#             # If LSTM is clearly best, give it more weight
#             if 'lstm' in good_models and lstm_r2 > 0.4:
#                 best_simple_r2 = max([v for k, v in good_models.items() if k != 'lstm'], default=0)
#                 if lstm_r2 > best_simple_r2 * 1.2:  # LSTM is 20% better
#                     # Boost LSTM weight
#                     self.weights['lstm'] = min(0.8, self.weights['lstm'] * 2)
                    
#                     # Renormalize other weights
#                     remaining_weight = 1.0 - self.weights['lstm']
#                     other_models = {k: v for k, v in self.weights.items() if k != 'lstm'}
#                     if other_models:
#                         other_total = sum(other_models.values())
#                         for k in other_models:
#                             self.weights[k] = remaining_weight * (other_models[k] / other_total)
        
#         # Store validation scores
#         self.validation_scores = model_scores
        
#         print("CORRECTED ensemble weights:", {k: f"{v:.4f}" for k, v in self.weights.items()})
        
#         # Validate ensemble performance
#         ensemble_pred = self.predict(X_val)
#         ensemble_r2 = r2_score(y_val, ensemble_pred)
#         best_individual_r2 = max(model_scores.values())
        
#         print(f"Ensemble validation R²: {ensemble_r2:.4f}")
#         print(f"Best individual R²: {best_individual_r2:.4f}")
        
#         # Safety check: If ensemble is much worse, fall back to best model
#         if ensemble_r2 < best_individual_r2 * 0.8:  # More than 20% worse
#             best_model = max(model_scores.items(), key=lambda x: x[1])
#             self.weights = {best_model[0]: 1.0}
#             print(f"⚠️  Ensemble underperforming. Falling back to best model: {best_model[0]}")
        
#     def predict(self, X_test):
#         """Make ensemble predictions using only weighted good models"""
#         predictions = {}
        
#         # LSTM prediction
#         if 'lstm' in self.weights:
#             lstm_pred = self.lstm_model.predict(X_test).flatten()
#             predictions['lstm'] = lstm_pred
        
#         # Simple model predictions
#         for model_name in self.simple_models.models.keys():
#             if model_name in self.weights:
#                 try:
#                     pred = self.simple_models.predict_single_model(model_name, X_test)
#                     predictions[model_name] = pred
#                 except:
#                     continue
        
#         # Weighted ensemble
#         if len(predictions) == 0:
#             # Fallback to LSTM if no predictions available
#             return self.lstm_model.predict(X_test).flatten()
        
#         # Calculate ensemble prediction
#         ensemble_pred = np.zeros_like(list(predictions.values())[0])
#         for model_name, weight in self.weights.items():
#             if model_name in predictions:
#                 ensemble_pred += weight * predictions[model_name]
        
#         return ensemble_pred
    
#     def evaluate(self, X_test, y_test):
#         """Evaluate ensemble performance"""
#         y_pred = self.predict(X_test)
        
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
#         rmse = np.sqrt(mse)
        
#         metrics = {
#             'MSE': mse,
#             'RMSE': rmse,
#             'MAE': mae,
#             'R2': r2
#         }
        
#         return metrics
    
#     def compare_models(self, X_test, y_test):
#         """Compare ensemble with individual models"""
#         results = {}
        
#         # Ensemble performance
#         results['ensemble'] = self.evaluate(X_test, y_test)
        
#         # LSTM performance
#         lstm_pred = self.lstm_model.predict(X_test).flatten()
#         results['lstm_only'] = {
#             'MSE': mean_squared_error(y_test, lstm_pred),
#             'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
#             'MAE': mean_absolute_error(y_test, lstm_pred),
#             'R2': r2_score(y_test, lstm_pred)
#         }
        
#         # Best performing simple models (only those with R² > 0.2)
#         for model_name in self.simple_models.models.keys():
#             try:
#                 pred = self.simple_models.predict_single_model(model_name, X_test)
#                 r2 = r2_score(y_test, pred)
#                 if r2 > 0.2:  # Only include decent models
#                     results[f'{model_name}'] = {
#                         'MSE': mean_squared_error(y_test, pred),
#                         'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
#                         'MAE': mean_absolute_error(y_test, pred),
#                         'R2': r2
#                     }
#             except:
#                 continue
                
#         return results


'''Trail 3'''

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from campus.decode_lstm import DecodeLSTM
from campus.simple_models import SimpleModels

class EnergyEnsemble:
    def __init__(self, input_shape):
        self.lstm_model = DecodeLSTM(input_shape)
        self.simple_models = SimpleModels()
        self.weights = {}
        self.validation_scores = {}
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train all models in the ensemble"""
        print("Training LSTM model...")
        self.lstm_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        print("Training simple models...")
        self.simple_models.train_models(X_train, y_train, X_val, y_val)
        
        print("Calculating LSTM-focused ensemble weights...")
        self.calculate_lstm_focused_weights(X_val, y_val)
        
    def calculate_lstm_focused_weights(self, X_val, y_val):
        """LSTM-focused ensemble weight calculation"""
        # Get predictions from all models
        lstm_pred = self.lstm_model.predict(X_val).flatten()
        lstm_r2 = r2_score(y_val, lstm_pred)
        
        print(f"LSTM R² on validation: {lstm_r2:.4f}")
        
        # Get simple model predictions and scores
        model_scores = {'lstm': lstm_r2}
        
        for model_name in self.simple_models.models.keys():
            try:
                pred = self.simple_models.predict_single_model(model_name, X_val)
                r2 = r2_score(y_val, pred)
                model_scores[model_name] = r2
                print(f"{model_name} R² on validation: {r2:.4f}")
            except:
                continue
        
        # **STRATEGY 1**: If LSTM is clearly best, make it dominant
        best_r2 = max(model_scores.values())
        second_best_r2 = sorted(model_scores.values(), reverse=True)[1] if len(model_scores) > 1 else 0
        
        if lstm_r2 >= best_r2 and lstm_r2 > 0.35:  # LSTM is best and good
            print(f"🎯 LSTM-Dominant Strategy: LSTM is best performer (R² = {lstm_r2:.4f})")
            
            # Give LSTM 70-80% weight
            lstm_weight = 0.75
            remaining_weight = 1.0 - lstm_weight
            
            # Find best simple model for complement
            simple_models = {k: v for k, v in model_scores.items() if k != 'lstm' and v > 0.25}
            
            if simple_models:
                # Give remaining weight to best simple model(s)
                best_simple = max(simple_models.items(), key=lambda x: x[1])
                
                if len(simple_models) == 1:
                    # Only one good simple model
                    self.weights = {
                        'lstm': lstm_weight,
                        best_simple[0]: remaining_weight
                    }
                else:
                    # Multiple good simple models - distribute remaining weight
                    good_simple = {k: v for k, v in simple_models.items() if v >= best_simple[1] * 0.8}
                    simple_total = sum(good_simple.values())
                    
                    self.weights = {'lstm': lstm_weight}
                    for model, score in good_simple.items():
                        self.weights[model] = remaining_weight * (score / simple_total)
            else:
                # No good simple models - use LSTM only
                self.weights = {'lstm': 1.0}
                print("No good simple models found. Using LSTM only.")
        
        # **STRATEGY 2**: Multiple competitive models
        elif len([r for r in model_scores.values() if r > 0.3]) >= 2:
            print("🔗 Multi-Model Strategy: Multiple competitive models found")
            
            # Only use models with R² > 0.3
            good_models = {k: v for k, v in model_scores.items() if v > 0.3}
            
            # Performance-based weights with LSTM bias
            weights = {}
            for model, r2 in good_models.items():
                if model == 'lstm':
                    weights[model] = r2 * 1.5  # 50% bonus for LSTM
                else:
                    weights[model] = r2
            
            # Normalize
            total_weight = sum(weights.values())
            self.weights = {k: w/total_weight for k, w in weights.items()}
        
        # **STRATEGY 3**: Fallback to best model only
        else:
            print("📊 Best-Model-Only Strategy: Using single best performer")
            best_model = max(model_scores.items(), key=lambda x: x[1])
            self.weights = {best_model[0]: 1.0}
        
        # Store validation scores
        self.validation_scores = model_scores
        
        print("LSTM-focused ensemble weights:", {k: f"{v:.4f}" for k, v in self.weights.items()})
        
        # **VALIDATION**: Test ensemble performance
        ensemble_pred = self.predict(X_val)
        ensemble_r2 = r2_score(y_val, ensemble_pred)
        
        print(f"Ensemble validation R²: {ensemble_r2:.4f}")
        print(f"LSTM validation R²: {lstm_r2:.4f}")
        
        # **SAFETY CHECK**: If ensemble is worse than LSTM, use LSTM only
        if ensemble_r2 < lstm_r2 * 0.95:  # Allow 5% tolerance
            print("⚠️  Ensemble underperforming LSTM. Switching to LSTM-only.")
            self.weights = {'lstm': 1.0}
            
            # Re-test
            ensemble_pred = self.predict(X_val)
            ensemble_r2 = r2_score(y_val, ensemble_pred)
            print(f"LSTM-only validation R²: {ensemble_r2:.4f}")
        
    def predict(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        
        # LSTM prediction
        if 'lstm' in self.weights:
            lstm_pred = self.lstm_model.predict(X_test).flatten()
            predictions['lstm'] = lstm_pred
        
        # Simple model predictions
        for model_name in self.simple_models.models.keys():
            if model_name in self.weights:
                try:
                    pred = self.simple_models.predict_single_model(model_name, X_test)
                    predictions[model_name] = pred
                except:
                    continue
        
        # Weighted ensemble
        if len(predictions) == 0:
            return self.lstm_model.predict(X_test).flatten()
        
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, weight in self.weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        
        return {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    def compare_models(self, X_test, y_test):
        """Compare ensemble with individual models"""
        results = {}
        
        # Ensemble performance
        results['ensemble'] = self.evaluate(X_test, y_test)
        
        # LSTM performance
        lstm_pred = self.lstm_model.predict(X_test).flatten()
        results['lstm_only'] = {
            'MSE': mean_squared_error(y_test, lstm_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
            'MAE': mean_absolute_error(y_test, lstm_pred),
            'R2': r2_score(y_test, lstm_pred)
        }
        
        # Only include decent simple models
        for model_name in self.simple_models.models.keys():
            try:
                pred = self.simple_models.predict_single_model(model_name, X_test)
                r2 = r2_score(y_test, pred)
                if r2 > 0.25:  # Only decent models
                    results[model_name] = {
                        'MSE': mean_squared_error(y_test, pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                        'MAE': mean_absolute_error(y_test, pred),
                        'R2': r2
                    }
            except:
                continue
                
        return results
