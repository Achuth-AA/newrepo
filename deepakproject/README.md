# IoT Sensor Anomaly Detection System

A comprehensive machine learning system for real-time anomaly detection in IoT sensor data, featuring multiple ML models, an interactive dashboard, and RESTful API.

## 📋 Project Overview

This project implements an end-to-end anomaly detection system for IoT sensor data from Intel Berkeley Research Lab. It includes:

- **Data Processing**: Cleaning, feature engineering, and preprocessing
- **Multiple ML Models**: Unsupervised (Isolation Forest, One-Class SVM, DBSCAN, Autoencoders) and Supervised (Random Forest, XGBoost, LSTM)
- **Hyperparameter Optimization**: Grid Search, Bayesian Optimization
- **REST API**: FastAPI backend with prediction endpoints
- **Interactive Dashboard**: Real-time monitoring with React + Tailwind CSS

## 📊 Dataset Information

- **Source**: Intel Berkeley Research Lab Sensor Data
- **Records**: 2,313,682 datapoints
- **Sensors**: 54 Mica2Dot sensors
- **Duration**: 36 days (February 28 - April 5, 2004)
- **Features**: timestamp, moteid, temperature, humidity, light, voltage

## 🏗️ Project Structure

```
deepakproject/
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration_and_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_unsupervised_anomaly_detection.ipynb
│   └── 04_supervised_models_with_tuning.ipynb
├── data/
│   ├── raw/                           # Raw data
│   └── processed/                     # Processed datasets
├── models/
│   └── saved_models/                  # Trained models
├── server/                            # FastAPI backend
│   └── main.py
├── client/                            # React frontend
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
├── archive (6)/                       # Original dataset
│   └── data.txt
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

#### 1. Clone the repository

```bash
cd C:\Users\achut\OneDrive\Desktop\deepakproject
```

#### 2. Set up Python environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Run Jupyter Notebooks (In Order)

Execute the notebooks in sequence to process data and train models:

```bash
jupyter notebook
```

1. **01_data_exploration_and_cleaning.ipynb** - Load and clean data
2. **02_feature_engineering.ipynb** - Create features
3. **03_unsupervised_anomaly_detection.ipynb** - Train unsupervised models
4. **04_supervised_models_with_tuning.ipynb** - Train supervised models with hyperparameter tuning

> **Important**: Run all notebooks in order before starting the backend server!

#### 4. Start the Backend Server

```bash
cd server
python main.py
```

The API will be available at `http://localhost:8000`

#### 5. Start the Frontend Dashboard

```bash
cd client
npm install
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## 📚 API Documentation

### Endpoints

#### Health Check
```
GET /
Returns API status and loaded models
```

#### Predictions

```
POST /predict/isolation_forest
POST /predict/random_forest
POST /predict/xgboost
POST /predict/autoencoder
POST /predict/ensemble
```

**Request Body:**
```json
{
  "readings": [
    {
      "timestamp": "2004-03-01 10:00:00",
      "moteid": 1,
      "temperature": 25.5,
      "humidity": 45.2,
      "light": 350.0,
      "voltage": 2.7
    }
  ]
}
```

**Response:**
```json
[
  {
    "timestamp": "2004-03-01 10:00:00",
    "moteid": 1,
    "is_anomaly": false,
    "anomaly_score": 0.123,
    "model": "ensemble",
    "confidence": 0.89
  }
]
```

#### Statistics
```
GET /stats
Returns overall system statistics and sensor health
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

## 🤖 Machine Learning Models

### Unsupervised Models

1. **Isolation Forest**
   - Fast anomaly detection
   - Works well with high-dimensional data
   - Contamination: 5%

2. **One-Class SVM**
   - Learns decision boundary around normal data
   - RBF kernel
   - Nu: 0.05

3. **DBSCAN**
   - Density-based clustering
   - Identifies outliers as noise points

4. **Autoencoder**
   - Deep learning approach
   - Reconstruction error-based detection
   - Architecture: 256-128-64-32-16 (encoder)

### Supervised Models

1. **Random Forest**
   - 300 estimators
   - Max depth: 30
   - Balanced class weights

2. **XGBoost**
   - Bayesian optimized hyperparameters
   - Scale pos weight for imbalance
   - Learning rate: 0.1

3. **LSTM**
   - Sequential pattern learning
   - 2 LSTM layers (128, 64 units)
   - Dropout: 0.3
   - Time steps: 10

### Ensemble Method

Combines predictions from multiple models using majority voting (2+ models agreeing).

## 📈 Feature Engineering

### Temporal Features
- Hour, day of week, cyclical encoding
- Time since start
- Weekend indicator

### Rolling Statistics
- Windows: 10, 30, 60 readings
- Mean, std, min, max

### Rate of Change
- First and second order differences
- Percentage change
- Deviation from rolling mean

### Lag Features
- Previous values (t-1, t-2, t-5, t-10)

### Inter-Sensor Features
- Global statistics
- Deviation from global mean
- Cross-sensor correlations

## 🎨 Dashboard Features

### Real-Time Monitoring
- Live anomaly detection
- Configurable detection models
- Recent anomalies feed

### Visualizations
- Time-series anomaly chart
- Sensor health grid (54 sensors)
- Statistics panels
- Model performance comparison

### Interactive Controls
- Model selector
- Time range filters
- Sensor filters
- Start/stop monitoring

## 🔧 Technologies Used

### Backend
- **FastAPI**: REST API framework
- **Scikit-learn**: Traditional ML models
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: Deep learning
- **Pandas/NumPy**: Data processing

### Frontend
- **React**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **JavaScript**: Programming language

## 📊 Model Performance

Performance metrics on test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| LSTM | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Ensemble | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

> **Note**: Run the notebooks to generate actual metrics.

## 🎯 Key Features

✅ **Robust Data Processing**
- Handles missing values
- Outlier detection and removal
- Feature normalization

✅ **Comprehensive Feature Engineering**
- 200+ engineered features
- Temporal patterns
- Statistical aggregations

✅ **Multiple ML Approaches**
- Unsupervised for exploration
- Supervised for classification
- Ensemble for best performance

✅ **Hyperparameter Optimization**
- GridSearchCV for Random Forest
- Bayesian Optimization for XGBoost
- Early stopping for deep learning

✅ **Production-Ready API**
- RESTful endpoints
- CORS enabled
- Model versioning
- Error handling

✅ **Interactive Dashboard**
- Real-time monitoring
- Multiple visualizations
- Responsive design
- Model selection

## 🚧 Challenges & Solutions

### Challenge 1: Class Imbalance
- **Solution**: SMOTE, class weights, ensemble methods

### Challenge 2: High Dimensionality
- **Solution**: PCA for DBSCAN, feature selection, robust models

### Challenge 3: Temporal Dependencies
- **Solution**: Time-series cross-validation, lag features, LSTM

### Challenge 4: Real-Time Processing
- **Solution**: Efficient data structures, model optimization, caching

## 📝 Future Improvements

- [ ] Add more visualization types (heatmaps, scatter plots)
- [ ] Implement user authentication
- [ ] Add model retraining pipeline
- [ ] Deploy to cloud (AWS/Azure/GCP)
- [ ] Add alerting system (email/SMS)
- [ ] Implement data versioning (DVC)
- [ ] Add A/B testing framework
- [ ] Create mobile app

## 👥 Team

- **Data Preprocessing**: [Name]
- **Model Development**: [Name]
- **Real-Time Application**: [Name]
- **Dashboard Creation**: [Name]
- **Documentation**: [Name]

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Intel Berkeley Research Lab for the dataset
- Open source community for the tools and libraries

## 📧 Contact

For questions or feedback, please contact [your-email@example.com]

---

**Generated with Claude Code** 🤖
