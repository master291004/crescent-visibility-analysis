# 🌙 Crescent Visibility Analysis and Prediction


## 📌 Project Overview

Predicting the visibility of the lunar crescent is a well-known astronomical and observational challenge with applications in calendar determination and observational astronomy.
This project analyzes a large dataset of lunar crescent observations and applies data analysis, geospatial visualization, and machine learning models to understand and predict crescent visibility based on astronomical and geographic features.

The repository is structured as a reproducible data science pipeline, separating data processing, visualization, modeling, and evaluation.

## 🎯 Objectives

* Perform exploratory data analysis (EDA) on lunar crescent visibility data

* Visualize geographic patterns of visibility across the world

* Engineer relevant astronomical features

* Train and evaluate machine learning models to predict crescent visibility

* Compare a baseline model (Logistic Regression) with a non-linear model (Random Forest)

* Identify the most influential features affecting visibility
## 🗂️ Project Structure
```
## 🗂️ Project Structure

```text
crescent-visibility-analysis/
│
├── data/
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── raw/
│       └── Final.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling_experiments.ipynb
│
├── src/
│   ├── data_processing.py      # Data loading, cleaning, feature engineering
│   ├── visualization.py        # Statistical and geospatial visualizations
│   ├── models.py               # Machine learning model training
│   ├── evaluation.py           # Model evaluation and metrics
│   ├── main.py                 # End-to-end pipeline execution
│   ├── __init__.py
│   └── data/
│       └── processed/
│
├── results/
│   ├── figures/
│   │   ├── map_all_visibility.png
│   │   ├── map_visibility_0.png
│   │   ├── map_visibility_1.png
│   │   ├── plot_distribution_Visibility.png
│   │   ├── plot_histograms.png
│   │   └── plot_correlation_heatmap.png
│   │
│   └── metrics/
│       ├── confusion_matrix.png
│       ├── confusion_matrix_lr.png
│       ├── roc_curve.png
│       ├── roc_curve_lr.png
│       ├── roc_comparison.png
│       └── model_comparison.png
│
├── requirements.txt
├── README.md
└── LICENSE
```
## 📊 Dataset Description
The dataset contains astronomical and geographical parameters, including:

* Latitude / Longitude

* Crescent width

* Arc of vision and arc of light

* Azimuth difference

* Elevation

* Time lag between sunset and moonset

* Visibility label (0 = Invisible, 1 = Visible)

Missing values are removed, time features are converted to numerical form, and the dataset is prepared for machine learning.

## 🔧 Feature Engineering

Key feature engineering steps include:

1. Conversion of sunset and moonset times into datetime format

2. Computation of lag in minutes between sunset and moonset

3. Encoding visibility labels into binary values

4. Renaming features for clarity and consistency

5. All processing steps are implemented in src/data_processing.py.

## 🌍 Visualization & Exploratory Analysis

The project includes:

* Distribution plots of visibility classes

* Histograms of numerical features

* Correlation heatmaps

* Geospatial maps showing crescent visibility across the world

* Separate maps for visible and invisible observations

* Visualizations are implemented in src/visualization.py and demonstrated in 03_visualization_examples.ipynb.

## 🤖 Machine Learning Models

Two supervised classification models are used:

### 1️⃣ Logistic Regression

* Serves as a baseline linear model

* Provides interpretability through coefficients

### 2️⃣ Random Forest Classifier

* Captures non-linear relationships

* Robust to feature scaling and noise

* Provides feature importance scores

* Models are trained and evaluated using:

* Train/Test split (80/20)

* Confusion Matrix

* Precision, Recall, F1-score

* ROC Curve and AUC

* Implementation is located in src/models.py and src/evaluation.py.

## 📈 Model Evaluation & Comparison

### Model performance is compared using:

* Accuracy

* Weighted F1-score

* ROC–AUC

Results demonstrate the advantage of non-linear models (Random Forest) in capturing complex astronomical relationships.

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/master291004/crescent-visibility-analysis.git
cd crescent-visibility-analysis
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** On Debian/Ubuntu systems, installing dependencies inside a virtual environment avoids PEP 668 ("externally managed environment") issues.

### 4. Run the Full Pipeline

```bash
python src/main.py
```

The pipeline will:

- Load and preprocess the crescent visibility dataset
- Perform feature engineering
- Generate geospatial and statistical visualizations
- Train Logistic Regression and Random Forest models
- Evaluate model performance
- Save all figures and metrics automatically

### 5. View Results

Generated outputs are saved to:

```text
results/
├── figures/
│   ├── map_all_visibility.png
│   ├── map_visibility_0.png
│   ├── map_visibility_1.png
│   ├── plot_distribution_Visibility.png
│   ├── plot_histograms.png
│   └── plot_correlation_heatmap.png
│
└── metrics/
    ├── confusion_matrix.png
    ├── confusion_matrix_lr.png
    ├── roc_curve.png
    ├── roc_curve_lr.png
    └── roc_comparison.png
```

### Cross-Platform Compatibility

The project uses Python's `pathlib` module for file handling, making it compatible with:

- Windows
- Linux
- macOS
- Docker environments
- CI/CD pipelines

No path modifications are required when running on different operating systems.
## 📚 Notebooks

* 01_data_exploration.ipynb – Exploratory analysis and initial insights

* 02_modeling_experiments.ipynb – Model training, tuning, and comparison


Notebooks are intended for analysis and presentation, while src/ contains reusable code.

## 🧠 Key Takeaways

Crescent visibility depends on multiple interacting astronomical factors

Geographical location plays a significant role

Non-linear models outperform linear baselines

Feature importance analysis highlights the most influential parameters

## 🚀 Future Work

* Hyperparameter tuning (GridSearch / RandomizedSearch)

* Cross-validation

* Additional models (XGBoost, SVM)

* Explainability tools (SHAP values)

* Temporal or seasonal analysis

## 👤 Author

Miled Trabelssi

Computer Engineering Student

Data Analysis & Machine Learning Enthusiast
