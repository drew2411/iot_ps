import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from generic.decode_lstm import DecodeLSTM
from generic.simple_models import SimpleModels

class EnergyEnsemble:
    def __init__(self, input_shape):
        self.lstm_model = DecodeLSTM(input_shape)
        self.simple_models = SimpleModels()
        self.weights = {}
        self.validation_scores = {}
        
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train all models in the ensemble"""
        print("Training LSTM model...")
        # Train LSTM
        self.lstm_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        print("Training simple models...")
        # Train simple models
        self.simple_models.train_models(X_train, y_train, X_val, y_val)
        
        print("Calculating ensemble weights...")
        # Calculate optimal weights based on validation performance
        self.calculate_weights(X_val, y_val)
        
    def calculate_weights(self, X_val, y_val):
        """Calculate ensemble weights based on validation performance"""
        # Get predictions from all models
        lstm_pred = self.lstm_model.predict(X_val).flatten()
        
        # Get simple model predictions
        simple_predictions = {}
        for model_name in self.simple_models.models.keys():
            try:
                pred = self.simple_models.predict_single_model(model_name, X_val)
                simple_predictions[model_name] = pred
            except:
                continue
        
        # Calculate individual model performance (inverse MSE for weights)
        scores = {}
        
        # LSTM score
        lstm_mse = mean_squared_error(y_val, lstm_pred)
        scores['lstm'] = 1.0 / (lstm_mse + 1e-6)
        
        # Simple model scores
        for model_name, pred in simple_predictions.items():
            mse = mean_squared_error(y_val, pred)
            scores[model_name] = 1.0 / (mse + 1e-6)
        
        # Normalize weights to sum to 1
        total_score = sum(scores.values())
        self.weights = {model: score/total_score for model, score in scores.items()}
        
        # Store validation scores
        self.validation_scores = scores
        
        print("Ensemble weights:", self.weights)
        
    def predict(self, X_test):
        """Make ensemble predictions"""
        predictions = {}
        
        # LSTM prediction
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
        ensemble_pred = np.zeros_like(lstm_pred)
        for model_name, weight in self.weights.items():
            if model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
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
        
        # Best simple model performance
        best_simple_score = -np.inf
        best_simple_name = None
        
        for model_name in self.simple_models.models.keys():
            try:
                pred = self.simple_models.predict_single_model(model_name, X_test)
                r2 = r2_score(y_test, pred)
                if r2 > best_simple_score:
                    best_simple_score = r2
                    best_simple_name = model_name
                    results[f'best_simple_{model_name}'] = {
                        'MSE': mean_squared_error(y_test, pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                        'MAE': mean_absolute_error(y_test, pred),
                        'R2': r2
                    }
            except:
                continue
                
        return results
