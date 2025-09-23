import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class EnergyOptimizer:
    def __init__(self, data, lstm_model, preprocessor):
        self.data = data
        self.lstm_model = lstm_model
        self.preprocessor = preprocessor
        self.optimization_results = {}
        self.recommendations = {}
        
    def analyze_consumption_patterns(self):
        """Analyze energy consumption patterns for optimization opportunities"""
        print("⚡ ENERGY CONSUMPTION PATTERN ANALYSIS")
        print("="*50)
        
        pattern_analysis = {}
        
        for building in self.data['building_type'].unique():
            building_data = self.data[self.data['building_type'] == building]
            
            if len(building_data) < 100:
                continue
            
            print(f"\nAnalyzing {building.upper()} consumption patterns...")
            
            # Daily patterns
            hourly_avg = building_data.groupby('hour')['energy_consumption'].agg(['mean', 'std'])
            daily_peak_hour = hourly_avg['mean'].idxmax()
            daily_low_hour = hourly_avg['mean'].idxmin()
            peak_consumption = hourly_avg['mean'].max()
            low_consumption = hourly_avg['mean'].min()
            
            # Weekly patterns
            weekly_avg = building_data.groupby('weekday')['energy_consumption'].agg(['mean', 'std'])
            peak_weekday = weekly_avg['mean'].idxmax()
            low_weekday = weekly_avg['mean'].idxmin()
            
            # Peak vs off-peak analysis
            peak_hours_data = building_data[building_data['is_peak_hour'] == 1]
            offpeak_hours_data = building_data[building_data['is_peak_hour'] == 0]
            
            peak_avg = peak_hours_data['energy_consumption'].mean()
            offpeak_avg = offpeak_hours_data['energy_consumption'].mean()
            peak_ratio = peak_avg / offpeak_avg if offpeak_avg > 0 else 1
            
            # Weekend vs weekday
            weekend_avg = building_data[building_data['weekend'] == 1]['energy_consumption'].mean()
            weekday_avg = building_data[building_data['weekend'] == 0]['energy_consumption'].mean()
            weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 1
            
            # Load factor analysis
            max_consumption = building_data['energy_consumption'].max()
            avg_consumption = building_data['energy_consumption'].mean()
            load_factor = avg_consumption / max_consumption if max_consumption > 0 else 0
            
            pattern_analysis[building] = {
                'daily_patterns': {
                    'peak_hour': daily_peak_hour,
                    'low_hour': daily_low_hour,
                    'peak_consumption': peak_consumption,
                    'low_consumption': low_consumption,
                    'daily_variation': (peak_consumption - low_consumption) / low_consumption if low_consumption > 0 else 0
                },
                'weekly_patterns': {
                    'peak_weekday': peak_weekday,
                    'low_weekday': low_weekday,
                    'weekend_ratio': weekend_ratio
                },
                'peak_analysis': {
                    'peak_avg': peak_avg,
                    'offpeak_avg': offpeak_avg,
                    'peak_ratio': peak_ratio
                },
                'efficiency_metrics': {
                    'load_factor': load_factor,
                    'avg_consumption': avg_consumption,
                    'max_consumption': max_consumption,
                    'consumption_variability': building_data['energy_consumption'].std() / avg_consumption
                }
            }
            
            print(f"  📈 Peak hour: {daily_peak_hour}, Low hour: {daily_low_hour}")
            print(f"  ⚖️ Peak/Off-peak ratio: {peak_ratio:.2f}")
            print(f"  📊 Load factor: {load_factor:.3f}")
            print(f"  🔄 Weekend/Weekday ratio: {weekend_ratio:.2f}")
        
        self.pattern_analysis = pattern_analysis
        return pattern_analysis
    
    def identify_optimization_opportunities(self):
        """Identify specific optimization opportunities"""
        print("\n🎯 OPTIMIZATION OPPORTUNITY IDENTIFICATION")
        print("="*50)
        
        if not hasattr(self, 'pattern_analysis'):
            self.analyze_consumption_patterns()
        
        opportunities = {}
        
        for building, patterns in self.pattern_analysis.items():
            building_opportunities = []
            potential_savings = 0
            
            print(f"\nOptimization opportunities for {building.upper()}:")
            
            # 1. Load Shifting Opportunities
            peak_ratio = patterns['peak_analysis']['peak_ratio']
            if peak_ratio > 1.5:
                load_shifting_potential = (peak_ratio - 1.2) * patterns['peak_analysis']['peak_avg'] * 0.3
                potential_savings += load_shifting_potential
                
                building_opportunities.append({
                    'type': 'Load Shifting',
                    'priority': 'High' if peak_ratio > 2.0 else 'Medium',
                    'description': f"Peak consumption is {peak_ratio:.1f}x higher than off-peak",
                    'potential_saving': load_shifting_potential,
                    'saving_percentage': ((peak_ratio - 1.2) * 0.3 / peak_ratio) * 100,
                    'implementation': 'Schedule non-critical loads during off-peak hours',
                    'target_hours': self.get_off_peak_hours(building)
                })
                print(f"  🔄 Load Shifting: {load_shifting_potential:.1f} units potential saving")
            
            # 2. Weekend Optimization
            weekend_ratio = patterns['weekly_patterns']['weekend_ratio']
            if weekend_ratio > 0.7 and building in ['academic', 'lecture', 'library']:
                weekend_saving = patterns['efficiency_metrics']['avg_consumption'] * weekend_ratio * 0.4
                potential_savings += weekend_saving
                
                building_opportunities.append({
                    'type': 'Weekend Schedule Optimization',
                    'priority': 'Medium',
                    'description': f"Weekend consumption is {weekend_ratio:.1f}x weekday levels",
                    'potential_saving': weekend_saving,
                    'saving_percentage': (weekend_ratio * 0.4) * 100,
                    'implementation': 'Implement automated weekend shutdown schedules',
                    'target_systems': ['HVAC', 'Lighting', 'Non-essential equipment']
                })
                print(f"  📅 Weekend Optimization: {weekend_saving:.1f} units potential saving")
            
            # 3. Load Factor Improvement
            load_factor = patterns['efficiency_metrics']['load_factor']
            if load_factor < 0.6:
                load_factor_improvement = patterns['efficiency_metrics']['max_consumption'] * (0.7 - load_factor) * 0.2
                potential_savings += load_factor_improvement
                
                building_opportunities.append({
                    'type': 'Load Factor Improvement',
                    'priority': 'Medium',
                    'description': f"Low load factor ({load_factor:.3f}) indicates inefficient usage",
                    'potential_saving': load_factor_improvement,
                    'saving_percentage': ((0.7 - load_factor) * 0.2) * 100,
                    'implementation': 'Redistribute loads and eliminate wasteful consumption',
                    'focus_areas': ['Equipment scheduling', 'Demand management']
                })
                print(f"  📊 Load Factor: {load_factor_improvement:.1f} units potential saving")
            
            # 4. Consumption Variability Reduction
            variability = patterns['efficiency_metrics']['consumption_variability']
            if variability > 0.5:
                variability_saving = patterns['efficiency_metrics']['avg_consumption'] * (variability - 0.3) * 0.15
                potential_savings += variability_saving
                
                building_opportunities.append({
                    'type': 'Consumption Stabilization',
                    'priority': 'Low',
                    'description': f"High consumption variability ({variability:.2f}) indicates unstable usage",
                    'potential_saving': variability_saving,
                    'saving_percentage': ((variability - 0.3) * 0.15) * 100,
                    'implementation': 'Implement consistent operational procedures',
                    'focus_areas': ['Equipment maintenance', 'Usage standardization']
                })
                print(f"  📉 Variability Reduction: {variability_saving:.1f} units potential saving")
            
            opportunities[building] = {
                'total_opportunities': len(building_opportunities),
                'opportunities': building_opportunities,
                'total_potential_saving': potential_savings,
                'total_saving_percentage': (potential_savings / patterns['efficiency_metrics']['avg_consumption']) * 100 if patterns['efficiency_metrics']['avg_consumption'] > 0 else 0
            }
            
            if potential_savings > 0:
                print(f"  💰 Total potential saving: {potential_savings:.1f} units ({opportunities[building]['total_saving_percentage']:.1f}%)")
            else:
                print(f"  ✅ Building operations appear optimized")
        
        self.optimization_opportunities = opportunities
        return opportunities
    
    def get_off_peak_hours(self, building):
        """Get recommended off-peak hours for load shifting"""
        if not hasattr(self, 'pattern_analysis'):
            return [22, 23, 0, 1, 2, 3, 4, 5, 6]  # Default off-peak hours
        
        building_data = self.data[self.data['building_type'] == building]
        hourly_avg = building_data.groupby('hour')['energy_consumption'].mean()
        
        # Find hours with consumption below 70% of peak
        peak_consumption = hourly_avg.max()
        off_peak_threshold = peak_consumption * 0.7
        
        off_peak_hours = hourly_avg[hourly_avg < off_peak_threshold].index.tolist()
        return off_peak_hours
    
    def generate_forecasting_based_optimization(self, building_type='academic', forecast_hours=24):
        """Generate optimization recommendations based on energy forecasting"""
        print(f"\n🔮 FORECASTING-BASED OPTIMIZATION FOR {building_type.upper()}")
        print("="*50)
        
        building_data = self.data[self.data['building_type'] == building_type]
        if len(building_data) == 0:
            print(f"No data available for {building_type}")
            return None
        
        # Get recent data for forecasting
        recent_data = building_data.tail(100).copy()  # Last 100 samples
        
        # Create future timestamps
        last_timestamp = recent_data['timestamp'].iloc[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=1),
            periods=forecast_hours * 60,  # Hourly forecasts in minutes
            freq='min'
        )
        
        # Generate simplified forecast (using last sequence pattern)
        feature_cols = self.preprocessor.create_feature_subsets()['all_features']
        
        try:
            # Create sequence for prediction
            last_sequence = recent_data[feature_cols].tail(10)
            
            # Simple forecast simulation (in practice, would use the actual model)
            base_pattern = recent_data.groupby('hour')['energy_consumption'].mean()
            forecasted_consumption = []
            
            for timestamp in future_timestamps[::60]:  # Hourly samples
                hour = timestamp.hour
                weekday = timestamp.weekday()
                is_weekend = 1 if weekday >= 5 else 0
                is_peak = 1 if (9 <= hour <= 11) or (14 <= hour <= 16) else 0
                
                # Base consumption from pattern
                base_consumption = base_pattern.get(hour, base_pattern.mean())
                
                # Apply modifiers
                if is_weekend:
                    weekend_modifier = self.pattern_analysis[building_type]['weekly_patterns']['weekend_ratio']
                    base_consumption *= weekend_modifier
                
                if is_peak:
                    peak_modifier = self.pattern_analysis[building_type]['peak_analysis']['peak_ratio']
                    base_consumption *= min(peak_modifier, 1.5)  # Cap the peak effect
                
                forecasted_consumption.append(base_consumption)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'timestamp': future_timestamps[::60],
                'hour': [t.hour for t in future_timestamps[::60]],
                'predicted_consumption': forecasted_consumption,
                'is_weekend': [1 if t.weekday() >= 5 else 0 for t in future_timestamps[::60]],
                'is_peak': [1 if (9 <= t.hour <= 11) or (14 <= t.hour <= 16) else 0 for t in future_timestamps[::60]]
            })
            
            # Identify optimization windows
            avg_consumption = np.mean(forecasted_consumption)
            
            # Low consumption periods (< 80% of average)
            low_consumption_periods = forecast_df[forecast_df['predicted_consumption'] < avg_consumption * 0.8]
            
            # High consumption periods (> 120% of average)  
            high_consumption_periods = forecast_df[forecast_df['predicted_consumption'] > avg_consumption * 1.2]
            
            optimization_windows = {
                'load_shifting_opportunities': {
                    'low_demand_hours': low_consumption_periods['hour'].unique().tolist(),
                    'high_demand_hours': high_consumption_periods['hour'].unique().tolist(),
                    'potential_shift_amount': (high_consumption_periods['predicted_consumption'].mean() - 
                                             low_consumption_periods['predicted_consumption'].mean()) * 0.2 if len(low_consumption_periods) > 0 and len(high_consumption_periods) > 0 else 0
                },
                'peak_shaving_opportunities': {
                    'peak_hours': high_consumption_periods[['timestamp', 'hour', 'predicted_consumption']].to_dict('records'),
                    'average_peak_reduction_potential': high_consumption_periods['predicted_consumption'].mean() * 0.15 if len(high_consumption_periods) > 0 else 0
                },
                'forecast_summary': {
                    'forecast_period': f"{forecast_hours} hours",
                    'avg_predicted_consumption': avg_consumption,
                    'peak_predicted_consumption': max(forecasted_consumption),
                    'min_predicted_consumption': min(forecasted_consumption),
                    'total_energy_forecast': sum(forecasted_consumption)
                }
            }
            
            print(f"Forecast generated for next {forecast_hours} hours")
            print(f"Average predicted consumption: {avg_consumption:.2f}")
            print(f"Peak reduction opportunities: {len(high_consumption_periods)} hours")
            print(f"Load shifting windows: {len(low_consumption_periods)} hours")
            
            return optimization_windows
            
        except Exception as e:
            print(f"Error generating forecast: {e}")
            return None
    
    def create_optimization_dashboard(self):
        """Create comprehensive optimization recommendations dashboard"""
        print("\n📋 OPTIMIZATION DASHBOARD GENERATION")
        print("="*50)
        
        if not hasattr(self, 'optimization_opportunities'):
            self.identify_optimization_opportunities()
        
        dashboard = {
            'executive_summary': {
                'total_buildings_analyzed': len(self.optimization_opportunities),
                'buildings_with_opportunities': sum(1 for opp in self.optimization_opportunities.values() 
                                                  if opp['total_opportunities'] > 0),
                'total_potential_savings': sum(opp['total_potential_saving'] 
                                             for opp in self.optimization_opportunities.values()),
                'average_saving_percentage': np.mean([opp['total_saving_percentage'] 
                                                    for opp in self.optimization_opportunities.values() 
                                                    if opp['total_saving_percentage'] > 0])
            },
            'priority_actions': [],
            'quick_wins': [],
            'long_term_projects': []
        }
        
        # Categorize recommendations by priority and effort
        all_opportunities = []
        for building, data in self.optimization_opportunities.items():
            for opp in data['opportunities']:
                opp['building'] = building
                all_opportunities.append(opp)
        
        # Sort by potential savings
        all_opportunities.sort(key=lambda x: x['potential_saving'], reverse=True)
        
        # Categorize
        for opp in all_opportunities:
            if opp['priority'] == 'High' and opp['potential_saving'] > 50:
                dashboard['priority_actions'].append(opp)
            elif opp['type'] in ['Weekend Schedule Optimization', 'Load Shifting'] and opp['potential_saving'] > 20:
                dashboard['quick_wins'].append(opp)
            else:
                dashboard['long_term_projects'].append(opp)
        
        print(f"Dashboard created:")
        print(f"  Priority actions: {len(dashboard['priority_actions'])}")
        print(f"  Quick wins: {len(dashboard['quick_wins'])}")
        print(f"  Long-term projects: {len(dashboard['long_term_projects'])}")
        print(f"  Total potential savings: {dashboard['executive_summary']['total_potential_savings']:.1f} units")
        
        self.optimization_dashboard = dashboard
        return dashboard
    
    def create_optimization_visualizations(self, save_path=None):
        """Create comprehensive optimization visualizations"""
        if not hasattr(self, 'optimization_opportunities'):
            self.identify_optimization_opportunities()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Energy Optimization Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Potential Savings by Building
        buildings = list(self.optimization_opportunities.keys())
        savings = [self.optimization_opportunities[b]['total_potential_saving'] for b in buildings]
        
        bars = axes[0,0].bar(buildings, savings, color='lightgreen', alpha=0.8)
        axes[0,0].set_title('Potential Energy Savings by Building')
        axes[0,0].set_ylabel('Potential Savings (Units)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, saving in zip(bars, savings):
            if saving > 0:
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(savings) * 0.01,
                              f'{saving:.1f}', ha='center', va='bottom')
        
        # 2. Optimization Opportunity Types Distribution
        all_types = []
        for building_data in self.optimization_opportunities.values():
            for opp in building_data['opportunities']:
                all_types.append(opp['type'])
        
        if all_types:
            type_counts = pd.Series(all_types).value_counts()
            axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0,1].set_title('Distribution of Optimization Opportunities')
        else:
            axes[0,1].text(0.5, 0.5, 'No optimization\nopportunities identified', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. Savings Percentage by Building
        saving_percentages = [self.optimization_opportunities[b]['total_saving_percentage'] for b in buildings]
        
        axes[0,2].bar(buildings, saving_percentages, color='orange', alpha=0.8)
        axes[0,2].set_title('Potential Savings Percentage by Building')
        axes[0,2].set_ylabel('Savings Percentage (%)')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% Target')
        axes[0,2].legend()
        
        # 4. Peak vs Off-Peak Ratios
        if hasattr(self, 'pattern_analysis'):
            peak_ratios = [self.pattern_analysis[b]['peak_analysis']['peak_ratio'] for b in buildings 
                          if b in self.pattern_analysis]
            building_names = [b for b in buildings if b in self.pattern_analysis]
            
            colors = ['red' if ratio > 2.0 else 'orange' if ratio > 1.5 else 'green' for ratio in peak_ratios]
            axes[1,0].bar(building_names, peak_ratios, color=colors, alpha=0.8)
            axes[1,0].set_title('Peak/Off-Peak Consumption Ratios')
            axes[1,0].set_ylabel('Peak/Off-Peak Ratio')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Optimization Threshold')
            axes[1,0].legend()
        
        # 5. Load Factors
        if hasattr(self, 'pattern_analysis'):
            load_factors = [self.pattern_analysis[b]['efficiency_metrics']['load_factor'] for b in buildings 
                           if b in self.pattern_analysis]
            
            colors = ['red' if lf < 0.5 else 'orange' if lf < 0.7 else 'green' for lf in load_factors]
            axes[1,1].bar(building_names, load_factors, color=colors, alpha=0.8)
            axes[1,1].set_title('Load Factor Analysis')
            axes[1,1].set_ylabel('Load Factor')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good Efficiency')
            axes[1,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Poor Efficiency')
            axes[1,1].legend()
        
        # 6. Implementation Priority Matrix
        if hasattr(self, 'optimization_dashboard'):
            priority_counts = {
                'High Priority': len(self.optimization_dashboard['priority_actions']),
                'Quick Wins': len(self.optimization_dashboard['quick_wins']),
                'Long-term': len(self.optimization_dashboard['long_term_projects'])
            }
            
            colors = ['red', 'orange', 'blue']
            axes[1,2].bar(priority_counts.keys(), priority_counts.values(), color=colors, alpha=0.8)
            axes[1,2].set_title('Implementation Priority Distribution')
            axes[1,2].set_ylabel('Number of Opportunities')
            axes[1,2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create hourly consumption patterns
        self.create_hourly_optimization_plot(save_path)
    
    def create_hourly_optimization_plot(self, save_path=None):
        """Create hourly consumption patterns with optimization recommendations"""
        if not hasattr(self, 'pattern_analysis'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hourly Consumption Patterns & Optimization Opportunities', fontsize=14, fontweight='bold')
        
        building_list = list(self.pattern_analysis.keys())[:4]  # Show up to 4 buildings
        
        for i, building in enumerate(building_list):
            row = i // 2
            col = i % 2
            
            building_data = self.data[self.data['building_type'] == building]
            hourly_pattern = building_data.groupby('hour')['energy_consumption'].agg(['mean', 'std'])
            
            # Plot mean consumption with error bars
            axes[row, col].errorbar(hourly_pattern.index, hourly_pattern['mean'], 
                                   yerr=hourly_pattern['std'], capsize=3, alpha=0.8, 
                                   label='Average ± Std')
            
            # Highlight optimization opportunities
            avg_consumption = hourly_pattern['mean'].mean()
            
            # High consumption hours (potential for reduction)
            high_hours = hourly_pattern[hourly_pattern['mean'] > avg_consumption * 1.2].index
            if len(high_hours) > 0:
                axes[row, col].fill_between(high_hours, 
                                           hourly_pattern.loc[high_hours, 'mean'], 
                                           alpha=0.3, color='red', 
                                           label='High Consumption (Reduce)')
            
            # Low consumption hours (potential for load shifting)
            low_hours = hourly_pattern[hourly_pattern['mean'] < avg_consumption * 0.8].index
            if len(low_hours) > 0:
                axes[row, col].fill_between(low_hours, 
                                           hourly_pattern.loc[low_hours, 'mean'], 
                                           alpha=0.3, color='green', 
                                           label='Low Consumption (Shift loads here)')
            
            axes[row, col].axhline(y=avg_consumption, color='blue', linestyle='--', 
                                  alpha=0.7, label='Average')
            axes[row, col].set_title(f'{building.title()} - Hourly Pattern')
            axes[row, col].set_xlabel('Hour of Day')
            axes[row, col].set_ylabel('Energy Consumption')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xticks(range(0, 24, 4))
        
        # Hide unused subplots
        for i in range(len(building_list), 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            hourly_path = save_path.replace('.png', '_hourly_optimization.png')
            plt.savefig(hourly_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_optimization_report(self, filepath):
        """Save comprehensive optimization report"""
        if not hasattr(self, 'optimization_opportunities'):
            self.identify_optimization_opportunities()
        
        if not hasattr(self, 'optimization_dashboard'):
            self.create_optimization_dashboard()
        
        with open(filepath, 'w') as f:
            f.write("CAMPUS ENERGY OPTIMIZATION REPORT\n")
            f.write("="*40 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*20 + "\n")
            summary = self.optimization_dashboard['executive_summary']
            f.write(f"Buildings analyzed: {summary['total_buildings_analyzed']}\n")
            f.write(f"Buildings with optimization opportunities: {summary['buildings_with_opportunities']}\n")
            f.write(f"Total potential savings: {summary['total_potential_savings']:.1f} units\n")
            f.write(f"Average potential savings: {summary.get('average_saving_percentage', 0):.1f}%\n\n")
            
            # Priority Actions
            f.write("PRIORITY ACTIONS (Immediate Implementation)\n")
            f.write("-"*45 + "\n")
            for i, action in enumerate(self.optimization_dashboard['priority_actions'], 1):
                f.write(f"\n{i}. {action['type']} - {action['building'].upper()}\n")
                f.write(f"   Description: {action['description']}\n")
                f.write(f"   Potential Saving: {action['potential_saving']:.1f} units ({action['saving_percentage']:.1f}%)\n")
                f.write(f"   Implementation: {action['implementation']}\n")
            
            # Quick Wins
            f.write(f"\nQUICK WINS (Easy Implementation)\n")
            f.write("-"*35 + "\n")
            for i, win in enumerate(self.optimization_dashboard['quick_wins'], 1):
                f.write(f"\n{i}. {win['type']} - {win['building'].upper()}\n")
                f.write(f"   Potential Saving: {win['potential_saving']:.1f} units\n")
                f.write(f"   Implementation: {win['implementation']}\n")
            
            # Detailed Building Analysis
            f.write(f"\nDETAILED BUILDING ANALYSIS\n")
            f.write("-"*30 + "\n")
            for building, data in self.optimization_opportunities.items():
                f.write(f"\n{building.upper()} BUILDING:\n")
                f.write(f"  Total Opportunities: {data['total_opportunities']}\n")
                f.write(f"  Potential Savings: {data['total_potential_saving']:.1f} units ({data['total_saving_percentage']:.1f}%)\n")
                
                for opp in data['opportunities']:
                    f.write(f"    • {opp['type']}: {opp['potential_saving']:.1f} units\n")
            
            # Implementation Timeline
            f.write(f"\nRECOMMENDED IMPLEMENTATION TIMELINE\n")
            f.write("-"*40 + "\n")
            f.write("Phase 1 (0-3 months): Priority Actions\n")
            for action in self.optimization_dashboard['priority_actions'][:3]:
                f.write(f"  • {action['building'].title()}: {action['type']}\n")
            
            f.write("\nPhase 2 (3-6 months): Quick Wins\n")
            for win in self.optimization_dashboard['quick_wins'][:3]:
                f.write(f"  • {win['building'].title()}: {win['type']}\n")
            
            f.write("\nPhase 3 (6-12 months): Long-term Projects\n")
            for project in self.optimization_dashboard['long_term_projects'][:3]:
                f.write(f"  • {project['building'].title()}: {project['type']}\n")
        
        print(f"Optimization report saved to: {filepath}")
