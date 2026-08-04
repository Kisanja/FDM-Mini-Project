# 🚗 AI-Powered Used Car Price Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Price%20Prediction-4CAF50)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

### An end-to-end machine learning platform for used-car price prediction, vehicle recommendations, market segmentation, and model explainability.

</div>

---

## 📌 Project Overview

The **AI-Powered Used Car Price Intelligence Platform** is an end-to-end data mining and machine learning project developed to analyse used-car market data and provide intelligent decision support.

The system combines data preprocessing, regression, clustering, recommendation logic, model diagnostics, and an interactive Streamlit interface in one integrated application.

Users can enter the specifications of a vehicle and receive:

- An estimated market price
- A market-segment classification
- Relevant vehicle recommendations
- Market insights and visualisations
- Explanations of the factors influencing model predictions

The project demonstrates the complete machine learning lifecycle, from data auditing and preprocessing to model training, artifact persistence, inference, and cloud deployment.

---

## 🎯 Project Objectives

The main objectives of this project are to:

1. Analyse and preprocess a large used-car dataset.
2. Identify the most important factors affecting vehicle prices.
3. Train a reliable machine learning model for used-car price prediction.
4. Segment vehicles into meaningful market groups using clustering.
5. recommend relevant vehicles based on user requirements.
6. Provide visual explanations and model-performance insights.
7. Deploy the completed solution as an interactive web application.

---

## ✨ Main Features

### 🏠 Home Dashboard

The home page provides an overview of the platform and summarises the available used-car data.

It includes information such as:

- Total number of vehicles
- Average market price
- Number of available vehicle brands
- Vehicle manufacturing-year range
- Quick navigation to all system functions

### 💰 Car Price Predictor

The price-prediction page allows users to enter vehicle specifications such as:

- Brand
- Model
- Manufacturing year
- Vehicle condition
- Mileage
- Engine size
- Fuel type
- Horsepower
- Torque
- Transmission
- Drive type
- Body type
- Accident history
- Fuel efficiency

The saved **LightGBM regression model** processes the information and returns an estimated used-car price.

### 🔍 Vehicle Recommendations

The recommendation module helps users discover vehicles that match their requirements.

Recommendations are generated using information such as:

- Preferred vehicle brand
- Vehicle type
- Price range
- Manufacturing year
- Mileage
- Fuel type
- Transmission
- Technical characteristics

The module uses the processed vehicle catalogue together with model predictions and filtering logic to return suitable options.

### 📊 Market Segmentation

The market-segmentation module groups vehicles with similar characteristics using **K-Means clustering**.

Typical clustering features include:

- Mileage
- Manufacturing year
- Horsepower
- Engine size

The generated clusters are converted into understandable market categories such as:

- Budget
- Mid-range
- Luxury

The module helps users understand where a selected vehicle is positioned within the used-car market.

### 🧠 Explainability and Model Insights

The explainability page presents visual information that helps users understand the machine learning model.

It can include:

- Important price-influencing features
- Model-performance visualisations
- Actual-versus-predicted price analysis
- Residual analysis
- Price distributions
- Market patterns
- Supporting model-development figures

This improves transparency and helps users understand why different vehicle characteristics affect the estimated price.

---

## 🧠 Machine Learning Approach

The project uses both supervised and unsupervised machine learning.

### Price Prediction

A **LightGBM Regressor** is used to estimate the market price of a vehicle.

LightGBM was selected because it performs effectively on structured tabular data and can capture non-linear relationships between vehicle specifications and prices.

### Market Segmentation

A **K-Means clustering model** is used to group vehicles with similar technical and market characteristics.

A saved feature scaler is applied before clustering to ensure that features with different measurement scales are treated appropriately.

### Recommendation System

The recommendation component uses the processed vehicle catalogue, user preferences, filtering logic, and predicted market information to identify relevant vehicles.

---

## 🔄 Machine Learning Workflow

```text
Raw Used-Car Dataset
        │
        ▼
Data Auditing and Exploration
        │
        ▼
Data Cleaning and Preprocessing
        │
        ├── Missing-value handling
        ├── Data-type correction
        ├── Outlier analysis
        ├── Feature engineering
        ├── Frequency encoding
        └── One-hot encoding
        │
        ▼
Model Development
        │
        ├── LightGBM Regression
        └── K-Means Clustering
        │
        ▼
Model Evaluation and Validation
        │
        ▼
Artifact Persistence
        │
        ├── Trained models
        ├── Feature metadata
        ├── Frequency mappings
        ├── Cluster metadata
        ├── Feature scaler
        └── Price-segment boundaries
        │
        ▼
Inference and Recommendation Layer
        │
        ▼
Streamlit Web Application
        │
        ▼
Streamlit Community Cloud Deployment
```

---

## 🧩 System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit User Interface                  │
│                                                              │
│  Home │ Price Predictor │ Recommendations │ Segments │ XAI   │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Application Service Layer                   │
│                                                              │
│  Input validation │ Session state │ Caching │ Page utilities │
└──────────────────────────────┬───────────────────────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
┌───────────────────────────┐   ┌──────────────────────────────┐
│      Inference Layer      │   │     Recommendation Layer     │
│                           │   │                              │
│ Feature construction      │   │ Catalogue loading            │
│ Price prediction          │   │ Preference filtering         │
│ Cluster assignment        │   │ Vehicle matching             │
│ Segment naming            │   │ Recommendation ranking       │
└──────────────┬────────────┘   └──────────────┬───────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Models and Data Artifacts                   │
│                                                              │
│ LightGBM │ K-Means │ Scaler │ Metadata │ Vehicle catalogue   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Programming and Data Processing

- Python
- Pandas
- NumPy
- SciPy

### Machine Learning

- LightGBM
- Scikit-learn
- Joblib
- K-Means clustering
- Feature scaling
- Regression modelling

### Data Visualisation

- Matplotlib
- Seaborn

### Web Application

- Streamlit

### Development and Deployment

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub
- Streamlit Community Cloud

---

## 📁 Project Structure

```text
FDM-Mini-Project/
│
├── app/
│   ├── Home.py
│   ├── _common.py
│   └── pages/
│       ├── 1_Car_Price_Predictor.py
│       ├── 2_Recommendations.py
│       ├── 3_Market_Segments.py
│       └── 4_Explainability.py
│
├── data/
│   └── Processed and application datasets
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── feature_columns.json
│   ├── model_freq_map.json
│   ├── kmeans.pkl
│   ├── kmeans_scaler.pkl
│   ├── kmeans_features.json
│   ├── kmeans_label_map.json
│   ├── price_bins.json
│   └── version.json
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_creation.ipynb
│   └── 04_smoke_tests.ipynb
│
├── reports/
│   └── Model figures, evaluation results, and supporting outputs
│
├── src/
│   ├── __init__.py
│   ├── inference.py
│   └── recommend.py
│
├── packages.txt
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 📦 Saved Model Artifacts

The application uses persisted model artifacts so that models do not need to be retrained every time the application starts.

| Artifact | Purpose |
|---|---|
| `lightgbm_model.pkl` | Trained used-car price-prediction model |
| `feature_columns.json` | Exact feature order required by the prediction model |
| `model_freq_map.json` | Frequency-encoding values for vehicle models |
| `kmeans.pkl` | Trained K-Means market-segmentation model |
| `kmeans_scaler.pkl` | Scaler used before K-Means prediction |
| `kmeans_features.json` | Features expected by the clustering model |
| `kmeans_label_map.json` | Human-readable names for cluster labels |
| `price_bins.json` | Price boundaries for Budget, Mid-range, and Luxury segments |
| `version.json` | Model and artifact version information |

---

## ⚙️ Installation and Local Setup

### Prerequisites

Install the following before running the project:

- Python 3.11
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Kisanja/FDM-Mini-Project.git
cd FDM-Mini-Project
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/Home.py
```

The application should open automatically in the browser.

The default local address is:

```text
http://localhost:8501
```

---

## 🧪 Testing the Saved Models

A basic inference test can be performed using the following code:

```python
from src.inference import load_artifacts, predict_and_cluster

artifacts = load_artifacts()

sample_vehicle = {
    "Brand": "Toyota",
    "Model": "Corolla",
    "Year": 2018,
    "Condition": "Used",
    "Mileage(km)": 84000,
    "EngineSize(L)": 1.6,
    "FuelType": "Gasoline",
    "Horsepower": 132,
    "Torque": 128,
    "Transmission": "Automatic",
    "DriveType": "FWD",
    "BodyType": "Sedan",
    "AccidentHistory": "No",
    "FuelEfficiency(L/100km)": 6.8,
}

result = predict_and_cluster(sample_vehicle, artifacts)

print(result)
```

The output contains:

```python
{
    "predicted_price": ...,
    "cluster_label": ...,
    "cluster_name": ...
}
```

---

## ☁️ Streamlit Community Cloud Deployment

The project is configured for deployment on Streamlit Community Cloud.

### Deployment Configuration

Use the following values when deploying:

```text
Main file path: app/Home.py
Python version: 3.11
```

The following files must remain in the repository root:

```text
requirements.txt
packages.txt
runtime.txt
```

### `runtime.txt`

```text
3.11
```

### `packages.txt`

```text
gfortran
liblapack-dev
libopenblas-dev
pkg-config
cmake
libgomp1
```

`libgomp1` provides the OpenMP runtime required by LightGBM in the cloud environment.

### Deployment Steps

1. Push the complete project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Choose the GitHub repository and deployment branch.
5. Set the main application file to `app/Home.py`.
6. Select Python `3.11` in the application settings.
7. Deploy the application.
8. Review the deployment logs to confirm that all dependencies and model artifacts were loaded.

---

## 🗂️ Important Deployment Requirements

Before deploying, confirm that the following files are committed to GitHub:

```text
models/lightgbm_model.pkl
models/feature_columns.json
models/model_freq_map.json
models/kmeans.pkl
models/kmeans_scaler.pkl
models/kmeans_features.json
models/kmeans_label_map.json
models/price_bins.json
```

The application cannot perform predictions when the required model files are missing from the deployed repository.

You can verify tracked model files using:

```bash
git ls-files models
```

---

## 🔧 Troubleshooting

### `ModuleNotFoundError: No module named 'joblib'`

Confirm that `joblib` is included in `requirements.txt`, then rebuild the Streamlit application.

### LightGBM model not found

Confirm that this file exists:

```text
models/lightgbm_model.pkl
```

Also ensure that it has been committed and pushed to GitHub.

```bash
git add models/
git commit -m "Add trained model artifacts"
git push
```

### Application uses the wrong Python version

Open:

```text
Manage app → Settings → General → Python version
```

Select:

```text
3.11
```

Save the changes and allow Streamlit Cloud to rebuild the application.

### LightGBM or OpenMP error

Confirm that `packages.txt` contains:

```text
libgomp1
```

### Old cached model or data is still being used

After deploying a new version:

```text
App menu → Clear cache → Clear caches
```

Then rerun or reboot the application.

### Models work locally but not in Streamlit Cloud

Check that:

- File names use the correct uppercase and lowercase letters.
- Model files are pushed to the deployed GitHub branch.
- The app is deployed from the correct repository and branch.
- The application is using Python 3.11.
- Required system packages are included in `packages.txt`.

---

## 🔐 Reliability Features

The inference layer includes several safeguards:

- Automatic repository-root detection
- Support for both `models/` and `app/models/`
- Validation of required artifacts
- Exact model-feature alignment
- Safe numeric conversion
- Missing clustering-value handling
- Joblib-based artifact loading
- Pickle fallback support
- Clear deployment error messages
- Streamlit caching for reusable models and data

---

## 📈 Future Improvements

Potential future developments include:

- More advanced hyperparameter optimisation
- Comparison with XGBoost, Random Forest, and CatBoost
- Confidence intervals for predicted prices
- More advanced similarity-based recommendation methods
- Integration with live vehicle-listing data
- Location-based price analysis
- Vehicle-price trend forecasting
- User accounts and saved recommendations
- REST API development for external applications
- Automated model retraining and versioning
- Docker-based deployment
- Additional model-explainability methods
- Mobile-responsive user-interface improvements

---

## ⚠️ Disclaimer

This system is an academic machine learning project.

Predicted prices and recommendations are estimates generated from historical data and model behaviour. Actual vehicle prices may vary according to location, market conditions, seller preferences, taxes, maintenance history, physical inspection results, and other external factors.

The application should not be treated as professional financial, valuation, or purchasing advice.

---

## 👥 Project Information

**Project:** AI-Powered Used Car Price Intelligence Platform  
**Project Type:** Data Mining and Machine Learning Mini Project  

---

## 🙏 Acknowledgements

We would like to thank our lecturers and academic supervisors for their guidance and feedback throughout this project.

We also acknowledge the developers and maintainers of Python, Streamlit, LightGBM, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and the other open-source technologies used in this system.

---

<div align="center">

### 🚗 Transforming used-car data into intelligent pricing decisions

Developed as an academic data mining and machine learning project.

</div>
