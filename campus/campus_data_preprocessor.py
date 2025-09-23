import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class CampusEnergyPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scalers = {}
        self.encoders = {}
        
    def load_all_buildings_data(self):
        """Load the main all_buildings_power.csv file"""
        filepath = os.path.join(self.dataset_path, 'all_buildings_power.csv')
        if os.path.exists(filepath):
            data = pd.read_csv(filepath)
            print(f"Loaded all_buildings_power.csv: {data.shape}")
            return data
        else:
            raise FileNotFoundError("all_buildings_power.csv not found!")
    
    def create_simplified_dataset(self):
        """Create simplified dataset focusing on power consumption only"""
        # Load main power consumption data
        power_data = self.load_all_buildings_data()
        
        # Convert Unix timestamp to datetime
        power_data['timestamp'] = pd.to_datetime(power_data['timestamp'], unit='s')
        power_data = power_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Date range: {power_data['timestamp'].min()} to {power_data['timestamp'].max()}")
        
        # Create unified dataset
        unified_data = []
        
        # Building mapping - focus on main buildings with good data
        building_columns = {
            'Academic': 'academic',
            'Boys_main': 'boys_main', 
            'Girls_main': 'girls_main',
            'Lecture': 'lecture',
            'Library': 'library'
        }
        
        for col, building_type in building_columns.items():
            if col in power_data.columns:
                # Extract power consumption for this building
                building_power = power_data[['timestamp', col]].copy()
                # Remove rows where power is NaN or 0
                building_power = building_power.dropna()
                building_power = building_power[building_power[col] > 0]
                
                if len(building_power) > 1000:  # Only include buildings with sufficient data
                    building_power['energy_consumption'] = building_power[col]
                    building_power['building_type'] = building_type
                    
                    # Add simple electrical parameters (estimated values)
                    building_power['voltage'] = 240.0 + np.random.normal(0, 5, len(building_power))
                    building_power['frequency'] = 50.0 + np.random.normal(0, 0.1, len(building_power))
                    building_power['power_factor'] = 0.95 + np.random.normal(0, 0.05, len(building_power))
                    
                    # Clip to realistic ranges
                    building_power['voltage'] = np.clip(building_power['voltage'], 220, 260)
                    building_power['frequency'] = np.clip(building_power['frequency'], 49, 51)
                    building_power['power_factor'] = np.clip(building_power['power_factor'], 0.8, 1.0)
                    
                    unified_data.append(building_power)
                    print(f"Added {building_type}: {len(building_power)} samples")
        
        # Combine all building data
        if unified_data:
            combined_data = pd.concat(unified_data, ignore_index=True)
            combined_data = combined_data.sort_values(['building_type', 'timestamp']).reset_index(drop=True)
            print(f"Combined dataset shape: {combined_data.shape}")
            return combined_data
        else:
            raise ValueError("No building data could be processed!")
    
    def add_temporal_features(self, data):
        """Add temporal features specific to campus energy patterns"""
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['weekday'] = data['timestamp'].dt.weekday
        data['weekend'] = (data['weekday'] >= 5).astype(int)
        
        # Campus-specific features
        data['is_working_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 18) & (data['weekday'] < 5)).astype(int)
        data['is_peak_hour'] = ((data['hour'] >= 9) & (data['hour'] <= 11) | 
                               (data['hour'] >= 14) & (data['hour'] <= 16)).astype(int)
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        return data
    
    def add_electrical_features(self, data):
        """Add derived electrical features"""
        # Power quality indicators
        data['voltage_deviation'] = abs(data['voltage'] - 240.0)
        data['frequency_deviation'] = abs(data['frequency'] - 50.0)
        data['low_power_factor'] = (data['power_factor'] < 0.9).astype(int)
        
        return data
    
    def add_lag_features_safe(self, data, target_col='energy_consumption'):
        """Add lag features safely with minimal data loss"""
        data = data.sort_values(['building_type', 'timestamp']).reset_index(drop=True)
        
        # Initialize lag columns
        lag_columns = [f'{target_col}_lag_1', f'{target_col}_lag_10', 
                      f'{target_col}_rolling_mean_10', f'{target_col}_rolling_std_10']
        for col in lag_columns:
            data[col] = np.nan
        
        for building in data['building_type'].unique():
            mask = data['building_type'] == building
            building_indices = data.index[mask]
            building_data = data.loc[mask, target_col]
            
            if len(building_data) > 20:  # Only process if sufficient data
                # Short-term lags (minimal data loss)
                data.loc[building_indices[1:], f'{target_col}_lag_1'] = building_data.iloc[:-1].values
                data.loc[building_indices[10:], f'{target_col}_lag_10'] = building_data.iloc[:-10].values
                
                # Rolling statistics
                rolling_mean = building_data.rolling(window=10, min_periods=5).mean()
                rolling_std = building_data.rolling(window=10, min_periods=5).std()
                
                data.loc[building_indices, f'{target_col}_rolling_mean_10'] = rolling_mean.values
                data.loc[building_indices, f'{target_col}_rolling_std_10'] = rolling_std.values
        
        return data
    
    def create_feature_subsets(self):
        """Define feature subsets for ensemble modeling"""
        temporal_features = [
            'hour', 'day', 'month', 'weekday', 'weekend',
            'is_working_hour', 'is_peak_hour', 'is_night',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]
        
        electrical_features = [
            'voltage', 'frequency', 'power_factor',
            'voltage_deviation', 'frequency_deviation', 'low_power_factor'
        ]
        
        lag_features = [
            'energy_consumption_lag_1', 'energy_consumption_lag_10',
            'energy_consumption_rolling_mean_10', 'energy_consumption_rolling_std_10'
        ]
        
        building_features = ['building_type_encoded']
        
        all_features = temporal_features + electrical_features + lag_features + building_features
        
        return {
            'temporal': temporal_features,
            'electrical': electrical_features,
            'lag_features': lag_features,
            'building': building_features,
            'all_features': all_features
        }
    
    def prepare_complete_dataset(self, target_col='energy_consumption', sample_size=100000):
        """Complete preprocessing pipeline with sampling for manageable size"""
        print("Creating simplified dataset...")
        data = self.create_simplified_dataset()
        
        # Sample data to manageable size for training
        if len(data) > sample_size:
            # Sample proportionally from each building
            sampled_data = []
            for building in data['building_type'].unique():
                building_data = data[data['building_type'] == building]
                building_sample_size = min(len(building_data), sample_size // len(data['building_type'].unique()))
                building_sample = building_data.sample(n=building_sample_size, random_state=42)
                sampled_data.append(building_sample)
            data = pd.concat(sampled_data, ignore_index=True)
            data = data.sort_values(['building_type', 'timestamp']).reset_index(drop=True)
            print(f"Sampled data to: {data.shape}")
        
        print("Adding temporal features...")
        data = self.add_temporal_features(data)
        
        print("Adding electrical features...")
        data = self.add_electrical_features(data)
        
        print("Adding lag features safely...")
        data = self.add_lag_features_safe(data, target_col)
        
        # Encode building types
        le = LabelEncoder()
        data['building_type_encoded'] = le.fit_transform(data['building_type'])
        self.encoders['building_type'] = le
        
        # Handle missing values more conservatively
        print("Handling missing values...")
        initial_length = len(data)
        
        # Fill missing values with forward fill, then backward fill, then mean
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != target_col:  # Don't fill target column
                data[col] = data.groupby('building_type')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill').fillna(x.mean())
                )
        
        # Only drop rows where target is missing
        data = data.dropna(subset=[target_col])
        
        # Drop rows with remaining NaN in critical features
        critical_features = ['energy_consumption_lag_1', target_col]
        data = data.dropna(subset=critical_features)
        
        print(f"Removed {initial_length - len(data)} rows with missing values")
        print(f"Final dataset shape: {data.shape}")
        
        if len(data) == 0:
            raise ValueError("All data was removed during preprocessing! Check your data quality.")
        
        return data
    
    def create_sequences(self, data, feature_cols, target_col, window_size=10):
        """Create sequences for LSTM with smaller window size"""
        # Reduce window size to prevent data loss
        all_X, all_y = [], []
        
        for building in data['building_type'].unique():
            building_data = data[data['building_type'] == building].copy()
            building_data = building_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(building_data) < window_size + 1:
                print(f"Skipping {building}: insufficient data ({len(building_data)} samples)")
                continue
                
            features = building_data[feature_cols].values
            target = building_data[target_col].values
            
            # Check for any remaining NaN values
            if np.isnan(features).any() or np.isnan(target).any():
                print(f"Warning: NaN values found in {building} data")
                continue
            
            # Scale features for this building
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers[f'features_{building}'] = scaler
            
            # Scale target
            target_scaler = StandardScaler()
            target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            self.scalers[f'target_{building}'] = target_scaler
            
            # Create sequences
            for i in range(window_size, len(features_scaled)):
                all_X.append(features_scaled[i-window_size:i])
                all_y.append(target_scaled[i])
            
            print(f"Created {len(features_scaled) - window_size} sequences for {building}")
        
        if len(all_X) == 0:
            raise ValueError("No sequences could be created! Check your data preprocessing.")
        
        return np.array(all_X), np.array(all_y)
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data maintaining temporal order"""
        total_samples = len(X)
        print(f"Total sequences available: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("No data to split!")
        
        # Calculate split indices
        train_end = int(total_samples * (1 - test_size - val_size))
        val_end = int(total_samples * (1 - test_size))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
