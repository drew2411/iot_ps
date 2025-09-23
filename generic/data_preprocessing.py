import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob
import os

class CampusEnergyPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.scalers = {}
        self.building_data = {}
        
    def load_campus_data(self):
        """Load all campus energy data files"""
        # Load individual building data
        building_files = {
            'academic': 'acad_build_mains.csv',
            'boys_hostel': 'boys_hostel_mains.csv',
            'girls_hostel': 'girls_hostel_mains.csv',
            'lecture': 'lecture_build_mains.csv',
            'library': 'library_build_mains.csv',
            'mess': 'mess_build_mains.csv',
            'facilities': 'facilities_build_mains.csv'
        }
        
        combined_data = []
        
        for building_type, filename in building_files.items():
            filepath = os.path.join(self.dataset_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['building_type'] = building_type
                df['building_id'] = building_type
                combined_data.append(df)
                print(f"Loaded {building_type}: {df.shape}")
        
        # Combine all building data
        if combined_data:
            campus_data = pd.concat(combined_data, ignore_index=True)
            print(f"Combined campus data shape: {campus_data.shape}")
            return campus_data
        else:
            print("No building data files found!")
            return None
    
    def explore_data_structure(self):
        """Explore the structure of your dataset"""
        # Load one file to check structure
        sample_file = os.path.join(self.dataset_path, 'all_buildings_power.csv')
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            print("Dataset columns:", df.columns.tolist())
            print("Dataset shape:", df.shape)
            print("First few rows:")
            print(df.head())
            print("\nData types:")
            print(df.dtypes)
            return df
        else:
            print("all_buildings_power.csv not found. Checking individual files...")
            # Try academic building file
            acad_file = os.path.join(self.dataset_path, 'acad_build_mains.csv')
            if os.path.exists(acad_file):
                df = pd.read_csv(acad_file)
                print("Sample file columns:", df.columns.tolist())
                print("Sample file shape:", df.shape)
                print("First few rows:")
                print(df.head())
                return df
    
    def create_unified_dataset(self):
        """Create a unified dataset for your ensemble model"""
        # Load the main combined file first
        main_file = os.path.join(self.dataset_path, 'all_buildings_power.csv')
        
        if os.path.exists(main_file):
            data = pd.read_csv(main_file)
        else:
            # If main file doesn't exist, combine individual files
            data = self.load_campus_data()
            if data is None:
                raise ValueError("Could not load any data files!")
        
        # Ensure timestamp column exists and is properly formatted
        timestamp_candidates = ['timestamp', 'datetime', 'date', 'time', 'Date', 'Time']
        timestamp_col = None
        
        for col in timestamp_candidates:
            if col in data.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            data['timestamp'] = pd.to_datetime(data[timestamp_col])
        else:
            print("No timestamp column found. Creating artificial timestamps...")
            data['timestamp'] = pd.date_range(start='2023-01-01', periods=len(data), freq='H')
        
        # Identify energy consumption column
        energy_candidates = ['power', 'energy', 'consumption', 'load', 'demand', 'Power', 'Energy']
        energy_col = None
        
        for col in energy_candidates:
            if col in data.columns:
                energy_col = col
                break
        
        if energy_col:
            data['energy_consumption'] = data[energy_col]
        else:
            print("Warning: No clear energy consumption column found!")
            print("Available columns:", data.columns.tolist())
            # Use first numeric column as energy consumption
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['energy_consumption'] = data[numeric_cols[0]]
                print(f"Using '{numeric_cols[0]}' as energy consumption")
        
        # Add building type if not present
        if 'building_type' not in data.columns:
            data['building_type'] = 'mixed_campus'
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
    
    def add_temporal_features(self, data):
        """Add temporal features for campus energy analysis"""
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['weekday'] = data['timestamp'].dt.weekday
        data['weekend'] = (data['weekday'] >= 5).astype(int)
        
        # Academic calendar features (useful for campus data)
        data['is_working_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 18) & (data['weekday'] < 5)).astype(int)
        data['is_peak_hour'] = ((data['hour'] >= 9) & (data['hour'] <= 11) | 
                               (data['hour'] >= 14) & (data['hour'] <= 16)).astype(int)
        
        # Cyclical encoding
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        return data
    
    def add_lag_features(self, data, target_col='energy_consumption'):
        """Add lag features specific to campus energy patterns"""
        # Sort by building and timestamp to ensure proper lag calculation
        data = data.sort_values(['building_type', 'timestamp']).reset_index(drop=True)
        
        # Create lag features within each building type
        for building in data['building_type'].unique():
            mask = data['building_type'] == building
            building_data = data.loc[mask, target_col]
            
            # 1-hour, 24-hour, and 168-hour (weekly) lags
            data.loc[mask, f'{target_col}_lag_1h'] = building_data.shift(1)
            data.loc[mask, f'{target_col}_lag_24h'] = building_data.shift(24)
            data.loc[mask, f'{target_col}_lag_168h'] = building_data.shift(168)
            
            # Rolling features
            data.loc[mask, f'{target_col}_rolling_mean_24h'] = building_data.rolling(window=24).mean()
            data.loc[mask, f'{target_col}_rolling_std_24h'] = building_data.rolling(window=24).std()
            data.loc[mask, f'{target_col}_rolling_mean_168h'] = building_data.rolling(window=168).mean()
        
        return data
    
    def prepare_for_ensemble(self, target_col='energy_consumption'):
        """Complete preprocessing pipeline for ensemble model"""
        print("Loading campus energy data...")
        data = self.create_unified_dataset()
        
        print("Adding temporal features...")
        data = self.add_temporal_features(data)
        
        print("Adding lag features...")
        data = self.add_lag_features(data, target_col)
        
        # Handle missing values
        data = data.fillna(method='forward').fillna(method='backward')
        
        # Remove rows with remaining NaN values
        initial_length = len(data)
        data = data.dropna()
        print(f"Removed {initial_length - len(data)} rows with missing values")
        
        # Define feature groups for ensemble
        temporal_features = ['hour', 'day', 'month', 'weekday', 'weekend', 
                           'is_working_hour', 'is_peak_hour',
                           'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        
        lag_features = [f'{target_col}_lag_1h', f'{target_col}_lag_24h', f'{target_col}_lag_168h',
                       f'{target_col}_rolling_mean_24h', f'{target_col}_rolling_std_24h', 
                       f'{target_col}_rolling_mean_168h']
        
        building_features = ['building_type']
        
        # Encode building types
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        data['building_type_encoded'] = le.fit_transform(data['building_type'])
        building_features = ['building_type_encoded']
        
        all_features = temporal_features + lag_features + building_features
        
        print(f"Final dataset shape: {data.shape}")
        print(f"Features for modeling: {len(all_features)}")
        
        return data, all_features

# Update main.py to use campus data
def main_campus():
    # Configuration
    DATASET_PATH = "D:\RHYTHM'S FOLDER\PS_IOT_2\energy_dataset"  # Update this path
    WINDOW_SIZE = 24
    TARGET_COLUMN = 'energy_consumption'
    
    print("Initializing Campus Energy Preprocessor...")
    preprocessor = CampusEnergyPreprocessor(DATASET_PATH)
    
    # First, explore the data structure
    print("Exploring data structure...")
    sample_data = preprocessor.explore_data_structure()
    
    # Prepare data for ensemble
    print("Preparing data for ensemble model...")
    data, feature_cols = preprocessor.prepare_for_ensemble(TARGET_COLUMN)
    
    # Continue with the rest of your ensemble pipeline...
    # (Use the same ensemble code from previous files)
    
    print("Campus energy analysis ready for ensemble modeling!")

if __name__ == "__main__":
    main_campus()
