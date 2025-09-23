import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

class BuildingEnergyAnalyzer:
    def __init__(self, data, lstm_model, preprocessor):
        self.data = data
        self.lstm_model = lstm_model
        self.preprocessor = preprocessor
        self.building_results = {}
        
    def analyze_building_performance(self, feature_cols, target_col='energy_consumption'):
        """Analyze LSTM performance across different building types"""
        print("🏢 MULTI-BUILDING PERFORMANCE ANALYSIS")
        print("="*50)
        
        for building in self.data['building_type'].unique():
            print(f"\nAnalyzing {building.upper()} building...")
            
            building_data = self.data[self.data['building_type'] == building].copy()
            building_data = building_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(building_data) < 100:
                print(f"  ⚠️ Insufficient data ({len(building_data)} samples)")
                continue
            
            try:
                # Create sequences for this building
                X_building, y_building = self.preprocessor.create_sequences(
                    building_data, feature_cols, target_col, window_size=10
                )
                
                if len(X_building) < 50:
                    print(f"  ⚠️ Insufficient sequences ({len(X_building)})")
                    continue
                
                # Use last 30% for testing
                test_size = int(len(X_building) * 0.3)
                X_test = X_building[-test_size:]
                y_test = y_building[-test_size:]
                
                # Get predictions
                predictions = self.lstm_model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, predictions.flatten())
                mae = mean_absolute_error(y_test, predictions.flatten())
                rmse = np.sqrt(mean_squared_error(y_test, predictions.flatten()))
                
                # Building-specific insights
                peak_consumption = building_data[building_data['is_peak_hour']==1]['energy_consumption'].mean()
                offpeak_consumption = building_data[building_data['is_peak_hour']==0]['energy_consumption'].mean()
                weekend_consumption = building_data[building_data['weekend']==1]['energy_consumption'].mean()
                weekday_consumption = building_data[building_data['weekend']==0]['energy_consumption'].mean()
                
                self.building_results[building] = {
                    'prediction_performance': {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'test_samples': len(X_test)
                    },
                    'consumption_patterns': {
                        'total_samples': len(building_data),
                        'avg_consumption': building_data['energy_consumption'].mean(),
                        'std_consumption': building_data['energy_consumption'].std(),
                        'peak_consumption': peak_consumption,
                        'offpeak_consumption': offpeak_consumption,
                        'peak_ratio': peak_consumption / offpeak_consumption if offpeak_consumption > 0 else 1,
                        'weekend_consumption': weekend_consumption,
                        'weekday_consumption': weekday_consumption,
                        'weekend_ratio': weekend_consumption / weekday_consumption if weekday_consumption > 0 else 1
                    },
                    'time_patterns': {
                        'peak_hour': building_data.groupby('hour')['energy_consumption'].mean().idxmax(),
                        'low_hour': building_data.groupby('hour')['energy_consumption'].mean().idxmin(),
                        'most_active_day': building_data.groupby('weekday')['energy_consumption'].mean().idxmax()
                    }
                }
                
                print(f"  ✅ R² = {r2:.4f}, MAE = {mae:.2f}")
                print(f"  📊 Peak/Off-peak ratio: {self.building_results[building]['consumption_patterns']['peak_ratio']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {building}: {e}")
        
        return self.building_results
    
    def generate_building_insights(self):
        """Generate actionable insights from building analysis"""
        insights = {
            'best_performing_building': None,
            'most_predictable': None,
            'highest_peak_ratio': None,
            'energy_efficiency_ranking': [],
            'recommendations': []
        }
        
        if not self.building_results:
            return insights
        
        # Find best performing building (prediction accuracy)
        best_r2 = -1
        for building, results in self.building_results.items():
            r2 = results['prediction_performance']['r2']
            if r2 > best_r2:
                best_r2 = r2
                insights['best_performing_building'] = building
        
        # Find most predictable (lowest MAE relative to consumption)
        lowest_relative_mae = float('inf')
        for building, results in self.building_results.items():
            mae = results['prediction_performance']['mae']
            avg_consumption = results['consumption_patterns']['avg_consumption']
            relative_mae = mae / avg_consumption if avg_consumption > 0 else float('inf')
            
            if relative_mae < lowest_relative_mae:
                lowest_relative_mae = relative_mae
                insights['most_predictable'] = building
        
        # Find building with highest peak ratio (most volatile)
        highest_ratio = 0
        for building, results in self.building_results.items():
            ratio = results['consumption_patterns']['peak_ratio']
            if ratio > highest_ratio:
                highest_ratio = ratio
                insights['highest_peak_ratio'] = building
        
        # Energy efficiency ranking (lower consumption per prediction accuracy)
        efficiency_scores = []
        for building, results in self.building_results.items():
            r2 = results['prediction_performance']['r2']
            avg_consumption = results['consumption_patterns']['avg_consumption']
            
            # Higher R² and lower consumption = more efficient
            efficiency_score = r2 / (avg_consumption / 1000)  # Normalize consumption
            efficiency_scores.append((building, efficiency_score))
        
        insights['energy_efficiency_ranking'] = sorted(efficiency_scores, key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        insights['recommendations'] = self.generate_recommendations()
        
        return insights
    
    def generate_recommendations(self):
        """Generate energy management recommendations"""
        recommendations = []
        
        for building, results in self.building_results.items():
            peak_ratio = results['consumption_patterns']['peak_ratio']
            weekend_ratio = results['consumption_patterns']['weekend_ratio']
            r2 = results['prediction_performance']['r2']
            
            # High peak ratio = load balancing opportunity
            if peak_ratio > 1.5:
                recommendations.append({
                    'building': building,
                    'type': 'Load Balancing',
                    'priority': 'High',
                    'recommendation': f"Consider load shifting for {building} - peak consumption is {peak_ratio:.1f}x higher than off-peak",
                    'potential_saving': f"{((peak_ratio - 1) * 0.2 * 100):.1f}% reduction possible"
                })
            
            # High weekend consumption = schedule optimization
            if weekend_ratio > 0.8 and building in ['academic', 'lecture', 'library']:
                recommendations.append({
                    'building': building,
                    'type': 'Schedule Optimization', 
                    'priority': 'Medium',
                    'recommendation': f"Review weekend operations for {building} - weekend consumption is {weekend_ratio:.1f}x weekday levels",
                    'potential_saving': f"{((weekend_ratio - 0.3) * 100):.1f}% weekend reduction possible"
                })
            
            # Low prediction accuracy = monitoring improvement
            if r2 < 0.3:
                recommendations.append({
                    'building': building,
                    'type': 'Monitoring Enhancement',
                    'priority': 'Medium',
                    'recommendation': f"Improve monitoring for {building} - energy patterns are less predictable (R² = {r2:.2f})",
                    'potential_saving': "Better monitoring enables 5-10% optimization"
                })
        
        return recommendations
    
    def create_building_visualizations(self, save_path=None):
        """Create comprehensive building analysis visualizations"""
        if not self.building_results:
            print("No building results to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Campus Building Energy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Prediction Performance by Building
        buildings = list(self.building_results.keys())
        r2_scores = [self.building_results[b]['prediction_performance']['r2'] for b in buildings]
        
        axes[0,0].bar(buildings, r2_scores, color='skyblue', alpha=0.8)
        axes[0,0].set_title('LSTM Prediction Accuracy by Building')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(r2_scores):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Average Consumption by Building
        avg_consumptions = [self.building_results[b]['consumption_patterns']['avg_consumption'] for b in buildings]
        
        axes[0,1].bar(buildings, avg_consumptions, color='lightcoral', alpha=0.8)
        axes[0,1].set_title('Average Energy Consumption by Building')
        axes[0,1].set_ylabel('Average Consumption')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Peak vs Off-Peak Ratios
        peak_ratios = [self.building_results[b]['consumption_patterns']['peak_ratio'] for b in buildings]
        
        axes[0,2].bar(buildings, peak_ratios, color='lightgreen', alpha=0.8)
        axes[0,2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Peak/Off-peak')
        axes[0,2].set_title('Peak/Off-Peak Consumption Ratio')
        axes[0,2].set_ylabel('Peak/Off-peak Ratio')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].legend()
        
        # 4. Weekend vs Weekday Ratios
        weekend_ratios = [self.building_results[b]['consumption_patterns']['weekend_ratio'] for b in buildings]
        
        axes[1,0].bar(buildings, weekend_ratios, color='gold', alpha=0.8)
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Weekend/Weekday')
        axes[1,0].set_title('Weekend/Weekday Consumption Ratio')
        axes[1,0].set_ylabel('Weekend/Weekday Ratio')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # 5. Prediction Error (MAE) by Building
        mae_scores = [self.building_results[b]['prediction_performance']['mae'] for b in buildings]
        
        axes[1,1].bar(buildings, mae_scores, color='orange', alpha=0.8)
        axes[1,1].set_title('Prediction Error (MAE) by Building')
        axes[1,1].set_ylabel('Mean Absolute Error')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Energy Efficiency Score (R² / Normalized Consumption)
        efficiency_scores = []
        for building in buildings:
            r2 = self.building_results[building]['prediction_performance']['r2']
            consumption = self.building_results[building]['consumption_patterns']['avg_consumption']
            efficiency = r2 / (consumption / 1000)  # Normalize
            efficiency_scores.append(efficiency)
        
        axes[1,2].bar(buildings, efficiency_scores, color='purple', alpha=0.8)
        axes[1,2].set_title('Energy Efficiency Score\n(Predictability/Consumption)')
        axes[1,2].set_ylabel('Efficiency Score')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create hourly pattern comparison
        self.create_hourly_pattern_plot(save_path)
    
    def create_hourly_pattern_plot(self, save_path=None):
        """Create hourly energy consumption patterns for all buildings"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        for building in self.data['building_type'].unique():
            building_data = self.data[self.data['building_type'] == building]
            hourly_pattern = building_data.groupby('hour')['energy_consumption'].mean()
            
            ax.plot(hourly_pattern.index, hourly_pattern.values, 
                   marker='o', linewidth=2, label=building.title(), alpha=0.8)
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Energy Consumption')
        ax.set_title('Daily Energy Consumption Patterns by Building Type')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 2))
        
        if save_path:
            pattern_path = save_path.replace('.png', '_hourly_patterns.png')
            plt.savefig(pattern_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_building_report(self, filepath):
        """Save comprehensive building analysis report"""
        insights = self.generate_building_insights()
        
        with open(filepath, 'w') as f:
            f.write("CAMPUS BUILDING ENERGY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            f.write(f"Buildings analyzed: {len(self.building_results)}\n")
            f.write(f"Best performing (prediction): {insights['best_performing_building']}\n")
            f.write(f"Most predictable: {insights['most_predictable']}\n")
            f.write(f"Highest peak ratio: {insights['highest_peak_ratio']}\n\n")
            
            # Detailed results
            f.write("DETAILED BUILDING ANALYSIS\n")
            f.write("-"*30 + "\n")
            
            for building, results in self.building_results.items():
                f.write(f"\n{building.upper()} BUILDING:\n")
                f.write(f"  Prediction Performance:\n")
                f.write(f"    R² Score: {results['prediction_performance']['r2']:.4f}\n")
                f.write(f"    MAE: {results['prediction_performance']['mae']:.2f}\n")
                f.write(f"    RMSE: {results['prediction_performance']['rmse']:.2f}\n")
                
                f.write(f"  Consumption Patterns:\n")
                f.write(f"    Average: {results['consumption_patterns']['avg_consumption']:.2f}\n")
                f.write(f"    Peak/Off-peak ratio: {results['consumption_patterns']['peak_ratio']:.2f}\n")
                f.write(f"    Weekend/Weekday ratio: {results['consumption_patterns']['weekend_ratio']:.2f}\n")
                
                f.write(f"  Time Patterns:\n")
                f.write(f"    Peak hour: {results['time_patterns']['peak_hour']}\n")
                f.write(f"    Low hour: {results['time_patterns']['low_hour']}\n")
            
            # Recommendations
            f.write(f"\nRECOMMENDATIONS\n")
            f.write("-"*15 + "\n")
            
            for i, rec in enumerate(insights['recommendations'], 1):
                f.write(f"\n{i}. {rec['type']} - {rec['building'].upper()} ({rec['priority']} Priority)\n")
                f.write(f"   {rec['recommendation']}\n")
                f.write(f"   Potential Impact: {rec['potential_saving']}\n")
        
        print(f"Building analysis report saved to: {filepath}")
