# crescent-visibility-analysis
Machine learning–based analysis and prediction of lunar crescent visibility using astronomical and geospatial features.
🌙 Crescent Visibility Analysis and Prediction
📌 Project Overview

Predicting the visibility of the lunar crescent is a well-known astronomical and observational challenge with applications in calendar determination and observational astronomy.
This project analyzes a large dataset of lunar crescent observations and applies data analysis, geospatial visualization, and machine learning models to understand and predict crescent visibility based on astronomical and geographic features.

The repository is structured as a reproducible data science pipeline, separating data processing, visualization, modeling, and evaluation.

🎯 Objectives

Perform exploratory data analysis (EDA) on lunar crescent visibility data

Visualize geographic patterns of visibility across the world

Engineer relevant astronomical features

Train and evaluate machine learning models to predict crescent visibility

Compare a baseline model (Logistic Regression) with a non-linear model (Random Forest)

Identify the most influential features affecting visibility

🗂️ Project Structure
crescent-visibility-analysis/
│
├── data/
│   ├── raw/                # Original dataset (Final.csv)
│   └── processed/          # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling_experiments.ipynb
│   └── 03_visualization_examples.ipynb
│
├── src/
│   ├── data_processing.py  # Data loading, cleaning, feature engineering
│   ├── visualization.py   # Plotting and geospatial visualizations
│   ├── models.py           # ML model training functions
│   ├── evaluation.py       # Metrics, ROC, confusion matrices
│   └── main.py             # End-to-end execution pipeline
│
├── results/
│   ├── figures/            # Generated plots and maps
│   └── metrics/            # Evaluation outputs (ROC, confusion matrix)
│
├── requirements.txt
└── README.md

📊 Dataset Description

The dataset contains astronomical and geographical parameters, including:

Latitude / Longitude

Crescent width

Arc of vision and arc of light

Azimuth difference

Elevation

Time lag between sunset and moonset

Visibility label (0 = Invisible, 1 = Visible)

Missing values are removed, time features are converted to numerical form, and the dataset is prepared for machine learning.

🔧 Feature Engineering

Key feature engineering steps include:

Conversion of sunset and moonset times into datetime format

Computation of lag in minutes between sunset and moonset

Encoding visibility labels into binary values

Renaming features for clarity and consistency

All processing steps are implemented in src/data_processing.py.

🌍 Visualization & Exploratory Analysis

The project includes:

Distribution plots of visibility classes

Histograms of numerical features

Correlation heatmaps

Geospatial maps showing crescent visibility across the world

Separate maps for visible and invisible observations

Visualizations are implemented in src/visualization.py and demonstrated in 03_visualization_examples.ipynb.

🤖 Machine Learning Models

Two supervised classification models are used:

1️⃣ Logistic Regression

Serves as a baseline linear model

Provides interpretability through coefficients

2️⃣ Random Forest Classifier

Captures non-linear relationships

Robust to feature scaling and noise

Provides feature importance scores

Models are trained and evaluated using:

Train/Test split (80/20)

Confusion Matrix

Precision, Recall, F1-score

ROC Curve and AUC

Implementation is located in src/models.py and src/evaluation.py.

📈 Model Evaluation & Comparison

Model performance is compared using:

Accuracy

Weighted F1-score

ROC–AUC

Results demonstrate the advantage of non-linear models (Random Forest) in capturing complex astronomical relationships.

▶️ How to Run the Project

Clone the repository:

git clone https://github.com/your-username/crescent-visibility-analysis.git
cd crescent-visibility-analysis


Install dependencies:

pip install -r requirements.txt


Run the full pipeline:

python src/main.py


All figures and evaluation metrics will be saved in the results/ directory.

📚 Notebooks

01_data_exploration.ipynb – Exploratory analysis and initial insights

02_modeling_experiments.ipynb – Model training, tuning, and comparison

03_visualization_examples.ipynb – Visual storytelling and result interpretation

Notebooks are intended for analysis and presentation, while src/ contains reusable code.

🧠 Key Takeaways

Crescent visibility depends on multiple interacting astronomical factors

Geographical location plays a significant role

Non-linear models outperform linear baselines

Feature importance analysis highlights the most influential parameters

🚀 Future Work

Hyperparameter tuning (GridSearch / RandomizedSearch)

Cross-validation

Additional models (XGBoost, SVM)

Explainability tools (SHAP values)

Temporal or seasonal analysis

👤 Author

Miled Trabelssi
Computer Engineering Student
Data Analysis & Machine Learning Enthusiast