import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

class EnergyAnomalyDetector:
    def __init__(self, data, lstm_model, preprocessor):
        self.data = data
        self.lstm_model = lstm_model
        self.preprocessor = preprocessor
        self.anomalies_detected = {}
        self.thresholds = {}
        
    def detect_prediction_anomalies(self, X_test, y_test, method='statistical'):
        """Detect anomalies using LSTM prediction errors"""
        print("🚨 ENERGY CONSUMPTION ANOMALY DETECTION")
        print("="*50)
        
        # Get predictions
        predictions = self.lstm_model.predict(X_test).flatten()
        
        # Calculate prediction errors
        absolute_errors = np.abs(y_test - predictions)
        relative_errors = absolute_errors / (np.abs(y_test) + 1e-8)  # Avoid division by zero
        
        if method == 'statistical':
            # Statistical method: Z-score based
            z_scores = np.abs(stats.zscore(absolute_errors))
            threshold = 3.0  # 3 standard deviations
            anomaly_mask = z_scores > threshold
            self.thresholds['statistical'] = threshold
            
        elif method == 'percentile':
            # Percentile method: 99th percentile
            threshold = np.percentile(absolute_errors, 99)
            anomaly_mask = absolute_errors > threshold
            self.thresholds['percentile'] = threshold
            
        elif method == 'iqr':
            # Interquartile Range method
            Q1 = np.percentile(absolute_errors, 25)
            Q3 = np.percentile(absolute_errors, 75)
            IQR = Q3 - Q1
            threshold = Q3 + 1.5 * IQR
            anomaly_mask = absolute_errors > threshold
            self.thresholds['iqr'] = threshold
        
        # Store results
        anomaly_results = {
            'method': method,
            'total_samples': len(y_test),
            'anomalies_detected': np.sum(anomaly_mask),
            'anomaly_rate': np.mean(anomaly_mask) * 100,
            'threshold': threshold,
            'anomaly_indices': np.where(anomaly_mask)[0],
            'anomaly_errors': absolute_errors[anomaly_mask],
            'avg_anomaly_magnitude': np.mean(absolute_errors[anomaly_mask]) if np.any(anomaly_mask) else 0,
            'max_anomaly_magnitude': np.max(absolute_errors[anomaly_mask]) if np.any(anomaly_mask) else 0
        }
        
        print(f"Method: {method}")
        print(f"Total samples: {anomaly_results['total_samples']}")
        print(f"Anomalies detected: {anomaly_results['anomalies_detected']}")
        print(f"Anomaly rate: {anomaly_results['anomaly_rate']:.2f}%")
        print(f"Threshold: {threshold:.4f}")
        print(f"Average anomaly magnitude: {anomaly_results['avg_anomaly_magnitude']:.4f}")
        
        self.anomalies_detected[method] = anomaly_results
        return anomaly_results, anomaly_mask
    
    def detect_building_specific_anomalies(self, feature_cols, target_col='energy_consumption'):
        """Detect anomalies for each building separately"""
        print("\n🏢 BUILDING-SPECIFIC ANOMALY DETECTION")
        print("="*50)
        
        building_anomalies = {}
        
        for building in self.data['building_type'].unique():
            print(f"\nAnalyzing anomalies in {building.upper()} building...")
            
            building_data = self.data[self.data['building_type'] == building].copy()
            building_data = building_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(building_data) < 100:
                print(f"  ⚠️ Insufficient data for {building}")
                continue
            
            try:
                # Create sequences
                X_building, y_building = self.preprocessor.create_sequences(
                    building_data, feature_cols, target_col, window_size=10
                )
                
                if len(X_building) < 50:
                    continue
                
                # Use all data for anomaly detection
                predictions = self.lstm_model.predict(X_building)
                errors = np.abs(y_building - predictions.flatten())
                
                # Statistical anomaly detection
                z_scores = np.abs(stats.zscore(errors))
                threshold = 2.5  # Slightly less strict for individual buildings
                anomaly_mask = z_scores > threshold
                
                # Temporal analysis of anomalies
                anomaly_indices = np.where(anomaly_mask)[0]
                if len(anomaly_indices) > 0:
                    # Map back to original timestamps (approximate)
                    start_idx = 10  # Window size offset
                    anomaly_timestamps = building_data.iloc[start_idx + anomaly_indices]['timestamp'].tolist()
                    anomaly_hours = building_data.iloc[start_idx + anomaly_indices]['hour'].tolist()
                    anomaly_days = building_data.iloc[start_idx + anomaly_indices]['weekday'].tolist()
                    
                    building_anomalies[building] = {
                        'total_samples': len(X_building),
                        'anomalies_count': np.sum(anomaly_mask),
                        'anomaly_rate': np.mean(anomaly_mask) * 100,
                        'threshold': threshold,
                        'anomaly_timestamps': anomaly_timestamps,
                        'anomaly_hours': anomaly_hours,
                        'anomaly_weekdays': anomaly_days,
                        'avg_anomaly_error': np.mean(errors[anomaly_mask]),
                        'max_anomaly_error': np.max(errors[anomaly_mask]),
                        'most_common_anomaly_hour': max(set(anomaly_hours), key=anomaly_hours.count) if anomaly_hours else None,
                        'most_common_anomaly_day': max(set(anomaly_days), key=anomaly_days.count) if anomaly_days else None
                    }
                    
                    print(f"  🔍 Found {np.sum(anomaly_mask)} anomalies ({np.mean(anomaly_mask)*100:.1f}%)")
                    print(f"  📅 Most common anomaly hour: {building_anomalies[building]['most_common_anomaly_hour']}")
                    print(f"  📅 Most common anomaly day: {building_anomalies[building]['most_common_anomaly_day']}")
                else:
                    building_anomalies[building] = {
                        'total_samples': len(X_building),
                        'anomalies_count': 0,
                        'anomaly_rate': 0,
                        'message': 'No significant anomalies detected'
                    }
                    print(f"  ✅ No significant anomalies detected")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing {building}: {e}")
        
        self.building_anomalies = building_anomalies
        return building_anomalies
    
    def detect_pattern_based_anomalies(self):
        """Detect anomalies based on consumption patterns"""
        print("\n📈 PATTERN-BASED ANOMALY DETECTION")
        print("="*50)
        
        pattern_anomalies = {}
        
        for building in self.data['building_type'].unique():
            building_data = self.data[self.data['building_type'] == building]
            
            if len(building_data) < 100:
                continue
            
            print(f"\nAnalyzing patterns in {building.upper()}...")
            
            # 1. Unusual consumption levels (compared to building's history)
            consumption = building_data['energy_consumption']
            consumption_z = np.abs(stats.zscore(consumption))
            consumption_anomalies = consumption_z > 3.0
            
            # 2. Unusual time-of-day consumption
            hourly_avg = building_data.groupby('hour')['energy_consumption'].mean()
            building_data_copy = building_data.copy()
            building_data_copy['expected_consumption'] = building_data_copy['hour'].map(hourly_avg)
            building_data_copy['hourly_deviation'] = (
                np.abs(building_data_copy['energy_consumption'] - building_data_copy['expected_consumption']) /
                building_data_copy['expected_consumption']
            )
            time_anomalies = building_data_copy['hourly_deviation'] > 2.0  # 200% deviation
            
            # 3. Weekend vs Weekday anomalies
            weekend_avg = building_data[building_data['weekend'] == 1]['energy_consumption'].mean()
            weekday_avg = building_data[building_data['weekend'] == 0]['energy_consumption'].mean()
            
            weekend_threshold = weekend_avg * 2.0  # Unusual weekend consumption
            weekday_threshold = weekday_avg * 0.3  # Unusual weekday low consumption
            
            weekend_anomalies = (
                (building_data['weekend'] == 1) & 
                (building_data['energy_consumption'] > weekend_threshold)
            )
            weekday_anomalies = (
                (building_data['weekend'] == 0) & 
                (building_data['energy_consumption'] < weekday_threshold)
            )
            
            # Combine all pattern anomalies
            all_pattern_anomalies = (
                consumption_anomalies | 
                time_anomalies | 
                weekend_anomalies | 
                weekday_anomalies
            )
            
            if np.any(all_pattern_anomalies):
                anomaly_data = building_data[all_pattern_anomalies]
                
                pattern_anomalies[building] = {
                    'total_samples': len(building_data),
                    'pattern_anomalies': np.sum(all_pattern_anomalies),
                    'anomaly_rate': np.mean(all_pattern_anomalies) * 100,
                    'consumption_anomalies': np.sum(consumption_anomalies),
                    'time_anomalies': np.sum(time_anomalies),
                    'weekend_anomalies': np.sum(weekend_anomalies),
                    'weekday_anomalies': np.sum(weekday_anomalies),
                    'anomaly_timestamps': anomaly_data['timestamp'].tolist(),
                    'anomaly_types': self.classify_anomaly_types(
                        consumption_anomalies, time_anomalies, 
                        weekend_anomalies, weekday_anomalies
                    )
                }
                
                print(f"  📊 Total pattern anomalies: {np.sum(all_pattern_anomalies)}")
                print(f"  🔸 Consumption level: {np.sum(consumption_anomalies)}")
                print(f"  🔸 Time-of-day: {np.sum(time_anomalies)}")
                print(f"  🔸 Weekend unusual: {np.sum(weekend_anomalies)}")
                print(f"  🔸 Weekday unusual: {np.sum(weekday_anomalies)}")
            else:
                pattern_anomalies[building] = {
                    'total_samples': len(building_data),
                    'pattern_anomalies': 0,
                    'message': 'No pattern anomalies detected'
                }
                print(f"  ✅ No pattern anomalies detected")
        
        self.pattern_anomalies = pattern_anomalies
        return pattern_anomalies
    
    def classify_anomaly_types(self, consumption, time_based, weekend, weekday):
        """Classify types of anomalies detected"""
        types = []
        if np.any(consumption):
            types.append('Extreme Consumption')
        if np.any(time_based):
            types.append('Unusual Time Pattern')
        if np.any(weekend):
            types.append('High Weekend Usage')
        if np.any(weekday):
            types.append('Low Weekday Usage')
        return types
    
    def generate_anomaly_insights(self):
        """Generate insights from anomaly detection"""
        insights = {
            'summary': {},
            'critical_buildings': [],
            'temporal_patterns': {},
            'recommendations': []
        }
        
        # Overall summary
        if hasattr(self, 'building_anomalies'):
            total_buildings = len(self.building_anomalies)
            buildings_with_anomalies = sum(1 for b in self.building_anomalies.values() 
                                         if b.get('anomalies_count', 0) > 0)
            
            insights['summary'] = {
                'total_buildings_analyzed': total_buildings,
                'buildings_with_anomalies': buildings_with_anomalies,
                'overall_anomaly_rate': np.mean([
                    b.get('anomaly_rate', 0) for b in self.building_anomalies.values()
                ])
            }
        
        # Identify critical buildings (high anomaly rates)
        if hasattr(self, 'building_anomalies'):
            for building, data in self.building_anomalies.items():
                if data.get('anomaly_rate', 0) > 5.0:  # More than 5% anomalies
                    insights['critical_buildings'].append({
                        'building': building,
                        'anomaly_rate': data['anomaly_rate'],
                        'common_hour': data.get('most_common_anomaly_hour'),
                        'common_day': data.get('most_common_anomaly_day')
                    })
        
        # Temporal patterns
        all_anomaly_hours = []
        all_anomaly_days = []
        
        if hasattr(self, 'building_anomalies'):
            for building, data in self.building_anomalies.items():
                if 'anomaly_hours' in data:
                    all_anomaly_hours.extend(data['anomaly_hours'])
                if 'anomaly_weekdays' in data:
                    all_anomaly_days.extend(data['anomaly_weekdays'])
        
        if all_anomaly_hours:
            insights['temporal_patterns'] = {
                'most_common_anomaly_hour': max(set(all_anomaly_hours), key=all_anomaly_hours.count),
                'most_common_anomaly_day': max(set(all_anomaly_days), key=all_anomaly_days.count) if all_anomaly_days else None,
                'peak_anomaly_hours': [h for h in set(all_anomaly_hours) if all_anomaly_hours.count(h) > len(all_anomaly_hours) * 0.1]
            }
        
        # Generate recommendations
        insights['recommendations'] = self.generate_anomaly_recommendations()
        
        return insights
    
    def generate_anomaly_recommendations(self):
        """Generate actionable recommendations based on anomalies"""
        recommendations = []
        
        if hasattr(self, 'building_anomalies'):
            for building, data in self.building_anomalies.items():
                anomaly_rate = data.get('anomaly_rate', 0)
                
                if anomaly_rate > 10:
                    recommendations.append({
                        'building': building,
                        'priority': 'Critical',
                        'type': 'System Investigation',
                        'recommendation': f"{building} shows {anomaly_rate:.1f}% anomaly rate - investigate equipment malfunctions or unusual usage patterns",
                        'action': 'Immediate technical inspection required'
                    })
                elif anomaly_rate > 5:
                    recommendations.append({
                        'building': building,
                        'priority': 'High',
                        'type': 'Monitoring Enhancement',
                        'recommendation': f"{building} has elevated anomaly rate ({anomaly_rate:.1f}%) - enhance monitoring systems",
                        'action': 'Install additional sensors or improve data collection'
                    })
                elif anomaly_rate > 2:
                    recommendations.append({
                        'building': building,
                        'priority': 'Medium',
                        'type': 'Usage Review',
                        'recommendation': f"{building} shows moderate anomalies ({anomaly_rate:.1f}%) - review usage patterns",
                        'action': 'Analyze occupancy schedules and equipment usage'
                    })
        
        # Pattern-based recommendations
        if hasattr(self, 'pattern_anomalies'):
            for building, data in self.pattern_anomalies.items():
                if data.get('weekend_anomalies', 0) > 10:
                    recommendations.append({
                        'building': building,
                        'priority': 'Medium',
                        'type': 'Schedule Optimization',
                        'recommendation': f"{building} has unusual weekend consumption patterns",
                        'action': 'Review weekend operations and implement automated scheduling'
                    })
        
        return recommendations
    
    def create_anomaly_visualizations(self, X_test, y_test, save_path=None):
        """Create comprehensive anomaly detection visualizations"""
        
        # Detect anomalies using multiple methods
        stat_results, stat_mask = self.detect_prediction_anomalies(X_test, y_test, 'statistical')
        perc_results, perc_mask = self.detect_prediction_anomalies(X_test, y_test, 'percentile')
        
        predictions = self.lstm_model.predict(X_test).flatten()
        errors = np.abs(y_test - predictions)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Energy Consumption Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Prediction vs Actual with Anomalies
        axes[0,0].scatter(y_test[~stat_mask], predictions[~stat_mask], alpha=0.6, s=20, 
                         color='blue', label='Normal')
        axes[0,0].scatter(y_test[stat_mask], predictions[stat_mask], alpha=0.8, s=30, 
                         color='red', label='Anomalies')
        
        min_val, max_val = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        axes[0,0].set_xlabel('Actual')
        axes[0,0].set_ylabel('Predicted')
        axes[0,0].set_title('Predictions with Anomalies Highlighted')
        axes[0,0].legend()
        
        # 2. Error Distribution with Threshold
        axes[0,1].hist(errors, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0,1].axvline(stat_results['threshold'], color='red', linestyle='--', 
                         label=f'Threshold: {stat_results["threshold"]:.3f}')
        axes[0,1].set_xlabel('Prediction Error')
        axes[0,1].set_ylabel('Density')
        axes[0,1].set_title('Error Distribution with Anomaly Threshold')
        axes[0,1].legend()
        
        # 3. Time Series with Anomalies
        sample_size = min(500, len(y_test))
        time_indices = range(sample_size)
        
        axes[0,2].plot(time_indices, y_test[:sample_size], label='Actual', alpha=0.7)
        axes[0,2].plot(time_indices, predictions[:sample_size], label='Predicted', alpha=0.7)
        
        # Highlight anomalies
        anomaly_indices = np.where(stat_mask[:sample_size])[0]
        if len(anomaly_indices) > 0:
            axes[0,2].scatter(anomaly_indices, y_test[anomaly_indices], 
                             color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)
        
        axes[0,2].set_xlabel('Time Index')
        axes[0,2].set_ylabel('Energy Consumption')
        axes[0,2].set_title('Time Series with Detected Anomalies')
        axes[0,2].legend()
        
        # 4. Anomaly Detection Methods Comparison
        methods = ['Statistical\n(Z-score)', 'Percentile\n(99th)']
        anomaly_counts = [stat_results['anomalies_detected'], perc_results['anomalies_detected']]
        
        bars = axes[1,0].bar(methods, anomaly_counts, color=['lightcoral', 'lightblue'], alpha=0.8)
        axes[1,0].set_title('Anomaly Detection Methods Comparison')
        axes[1,0].set_ylabel('Number of Anomalies')
        
        # Add value labels on bars
        for bar, count in zip(bars, anomaly_counts):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                          str(count), ha='center', va='bottom')
        
        # 5. Error Magnitude Analysis
        if np.any(stat_mask):
            normal_errors = errors[~stat_mask]
            anomaly_errors = errors[stat_mask]
            
            axes[1,1].boxplot([normal_errors, anomaly_errors], 
                             labels=['Normal', 'Anomalies'], patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red'))
            axes[1,1].set_ylabel('Prediction Error')
            axes[1,1].set_title('Error Magnitude: Normal vs Anomalies')
        
        # 6. Anomaly Rate by Building (if available)
        if hasattr(self, 'building_anomalies'):
            buildings = list(self.building_anomalies.keys())
            rates = [self.building_anomalies[b].get('anomaly_rate', 0) for b in buildings]
            
            if buildings and any(r > 0 for r in rates):
                axes[1,2].bar(buildings, rates, color='orange', alpha=0.8)
                axes[1,2].set_title('Anomaly Rate by Building')
                axes[1,2].set_ylabel('Anomaly Rate (%)')
                axes[1,2].tick_params(axis='x', rotation=45)
                
                # Add threshold line
                axes[1,2].axhline(y=5.0, color='red', linestyle='--', alpha=0.7, 
                                 label='5% Threshold')
                axes[1,2].legend()
        else:
            axes[1,2].text(0.5, 0.5, 'Building-specific\nanalysis not available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Building-Specific Anomalies')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_anomaly_report(self, filepath):
        """Save comprehensive anomaly detection report"""
        insights = self.generate_anomaly_insights()
        
        with open(filepath, 'w') as f:
            f.write("ENERGY CONSUMPTION ANOMALY DETECTION REPORT\n")
            f.write("="*55 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            if insights['summary']:
                f.write(f"Buildings analyzed: {insights['summary']['total_buildings_analyzed']}\n")
                f.write(f"Buildings with anomalies: {insights['summary']['buildings_with_anomalies']}\n")
                f.write(f"Average anomaly rate: {insights['summary']['overall_anomaly_rate']:.2f}%\n")
            
            f.write(f"Critical buildings: {len(insights['critical_buildings'])}\n\n")
            
            # Detection Results
            if hasattr(self, 'anomalies_detected'):
                f.write("ANOMALY DETECTION RESULTS\n")
                f.write("-"*30 + "\n")
                for method, results in self.anomalies_detected.items():
                    f.write(f"\n{method.upper()} METHOD:\n")
                    f.write(f"  Total samples: {results['total_samples']}\n")
                    f.write(f"  Anomalies detected: {results['anomalies_detected']}\n")
                    f.write(f"  Anomaly rate: {results['anomaly_rate']:.2f}%\n")
                    f.write(f"  Threshold: {results['threshold']:.4f}\n")
                    f.write(f"  Average magnitude: {results['avg_anomaly_magnitude']:.4f}\n")
            
            # Building-Specific Results
            if hasattr(self, 'building_anomalies'):
                f.write(f"\nBUILDING-SPECIFIC ANOMALIES\n")
                f.write("-"*30 + "\n")
                for building, data in self.building_anomalies.items():
                    f.write(f"\n{building.upper()}:\n")
                    if 'anomalies_count' in data:
                        f.write(f"  Anomalies: {data['anomalies_count']} ({data['anomaly_rate']:.1f}%)\n")
                        if data.get('most_common_anomaly_hour'):
                            f.write(f"  Most common hour: {data['most_common_anomaly_hour']}\n")
                        if data.get('most_common_anomaly_day'):
                            f.write(f"  Most common day: {data['most_common_anomaly_day']}\n")
                    else:
                        f.write(f"  {data.get('message', 'No anomalies')}\n")
            
            # Temporal Patterns
            if insights['temporal_patterns']:
                f.write(f"\nTEMPORAL PATTERNS\n")
                f.write("-"*20 + "\n")
                f.write(f"Most common anomaly hour: {insights['temporal_patterns']['most_common_anomaly_hour']}\n")
                if insights['temporal_patterns']['most_common_anomaly_day']:
                    f.write(f"Most common anomaly day: {insights['temporal_patterns']['most_common_anomaly_day']}\n")
            
            # Critical Buildings
            if insights['critical_buildings']:
                f.write(f"\nCRITICAL BUILDINGS (>5% anomaly rate)\n")
                f.write("-"*40 + "\n")
                for building in insights['critical_buildings']:
                    f.write(f"{building['building'].upper()}: {building['anomaly_rate']:.1f}% anomaly rate\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS\n")
            f.write("-"*15 + "\n")
            for i, rec in enumerate(insights['recommendations'], 1):
                f.write(f"\n{i}. {rec['type']} - {rec['building'].upper()} ({rec['priority']} Priority)\n")
                f.write(f"   Issue: {rec['recommendation']}\n")
                f.write(f"   Action: {rec['action']}\n")
        
        print(f"Anomaly detection report saved to: {filepath}")
