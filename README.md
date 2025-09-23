# Intelligent IoT Data Analysis for Smart Energy Management

## 🏢 Multi-Building Campus Energy Management System

A comprehensive IoT-based energy management system that combines LSTM-based prediction, anomaly detection, and optimization for multi-building campus environments.

## 🎯 Key Features

- **Multi-Building Analysis**: Comparative energy prediction across 5 building types
- **LSTM-Based Forecasting**: Achieved R² scores of 0.32-0.487 across different buildings
- **Anomaly Detection**: Statistical and pattern-based anomaly identification
- **Energy Optimization**: Quantified energy-saving recommendations (15-30% potential savings)
- **Real Campus IoT Data**: Validated on actual university sensor data (2.3M+ data points)

## 📊 Performance Results

| Building Type | R² Score | RMSE | MAE |
|---------------|----------|------|-----|
| Boys Hostel   | 0.487    | 0.74 | 0.52|
| Girls Hostel  | 0.482    | 0.74 | 0.52|
| Library       | 0.418    | 0.79 | 0.58|
| Academic      | 0.410    | 0.81 | 0.60|
| Lecture Halls | 0.322    | 0.85 | 0.62|

## 🚀 Quick Start

### Prerequisites
Python 3.8+
pip install -r requirements.txt

### Installation

git clone https://github.com/yourusername/IoT-Campus-Energy-Management.git
cd IoT-Campus-Energy-Management
pip install -r requirements.txt

### Usage

#### Run Campus-Specific Analysis:


## 📁 Project Structure

- **`campus_implementation/`**: Campus-specific analysis modules
- **`generic_implementation/`**: Generic DECODE reproduction
- **`results/`**: Generated reports and visualizations
- **`docs/`**: Documentation and architecture diagrams

## 🔧 System Architecture

![System Architecture](docs/system_architecture.png)

## 📈 Key Contributions

1. **Multi-Building Comparative Analysis**: First comprehensive campus-wide energy analysis
2. **Real IoT Data Validation**: Practical deployment with actual sensor data
3. **Integrated Framework**: Combined prediction, anomaly detection, and optimization
4. **Building-Specific Insights**: Tailored recommendations per building type

## 📊 Modules

### 🏢 Building Analysis
- Multi-building performance comparison
- Energy pattern discovery
- Efficiency ranking system

### 🚨 Anomaly Detection
- Statistical anomaly detection (Z-score, percentile)
- Pattern-based anomaly identification
- Building-specific anomaly analysis

### ⚡ Energy Optimization
- Load shifting recommendations
- Peak/off-peak analysis
- Quantified energy savings calculation

## 🛠️ Technical Details

- **Deep Learning**: 2-layer LSTM with dropout regularization
- **Feature Engineering**: 36 features across temporal, electrical, and building domains
- **Data Processing**: Handles 2.3M+ IoT sensor readings
- **Real-Time Capable**: Streaming data analysis ready

## 📋 Requirements

See `requirements.txt` for complete dependencies.

## 📚 Related Work

This project extends the DECODE framework for multi-building campus environments:
- DECODE: Data-driven Energy Consumption Prediction leveraging Historical Data and Environmental Factors

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🎓 Author

**Your Name** - Undergraduate Research Project  
University Name - Department  
Contact: your.email@university.edu

## 🙏 Acknowledgments

- DECODE paper authors for the foundational methodology
- University facilities team for providing campus IoT data
- Research supervisors and advisors

---

⭐ If you find this project useful, please consider giving it a star!
