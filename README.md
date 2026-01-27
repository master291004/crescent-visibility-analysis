# crescent-visibility-analysis
Machine learning–based analysis and prediction of lunar crescent visibility using astronomical and geospatial features.
### crescent-visibility-analysis
This project analyzes lunar crescent visibility data and builds machine learning models to predict whether the crescent will be visible from a given location and time.
The goal is to combine geospatial analysis, feature engineering, and predictive modeling to understand patterns in lunar observations.

## Dataset

The dataset contains observations of crescent visibility across multiple locations and dates. Key features include:

## Feature	Description
Lat, Long	Geographic coordinates of observation
ArcV, ArcL	Arc of Vision and Arc of Light
DAZ	Difference in azimuth between moon and sun
W	Crescent width (in degrees)
JD	Julian Date of observation
Ele	Elevation angle
Sunset, Moonset	Times of sunset and moonset
V	Visibility label (V = visible, I = invisible)

The raw data is located in data/raw/Final.csv. After processing, cleaned data is stored in data/processed/processed_data.csv.

## Project Structure 

crescent-visibility-analysis/
├── data/
│   ├── raw/                 # Raw dataset
│   └── processed/           # Cleaned dataset
├── notebooks/               # Exploration and visualization notebooks
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling_experiments.ipynb
├── results/
│   ├── figures/            # Generated plots and maps
│   └── metrics/          
├── src/                     # Reproducible Python modules
│   ├── data_processing.py   # Loading, cleaning, feature engineering
│   ├── models.py            # Model training and prediction
│   └── evaluation.py        # Evaluation functions (metrics, ROC, confusion matrix)
├── README.md
└── requirements.txt         # Python dependencies
## Workflow 
# Data Processing (src/data_processing.py)

Load raw data

Handle missing values

Compute Lag_minutes (Moonset - Sunset)

Encode Visibility as binary

Save cleaned data

# Exploratory Data Analysis (notebooks/01_data_exploration.ipynb)

Feature distributions and histograms

Geospatial maps of visibility (all points, visible only, invisible only)

Scatter plots of key features


Correlation analysis

Initial observations for modeling

# Modeling Experiments (notebooks/02_modeling_experiments.ipynb)

Train baseline and Random Forest models

Evaluate metrics: accuracy, precision, recall, F1-score, ROC AUC

Feature importance analysis

Optional hyperparameter tuning

## Key Findings 

# Random Forest  
outperforms baseline models (e.g., Logistic Regression) for predicting visibility.

# Lag_minutes and Crescent Width
 are the most important features for predicting visibility.

Visible crescent observations tend to cluster in specific latitudes and regions, as shown in geospatial maps.

Feature distributions and correlations suggest non-linear relationships that Random Forest can capture effectively.

## How to Run

1. Clone the repository :

git clone https://github.com/yourusername/crescent-visibility-analysis.git
cd crescent-visibility-analysis

2. Install dependencies :

pip install -r requirements.txt

3. Run the main pipeline :

python src/main.py

4. Open notebooks for exploration and visualization:

jupyter notebook

## Visualizations

# Distribution of Visibility: 
Shows the number of visible vs invisible crescent observations

# Feature Histograms:
 Understand feature ranges and distributions

# Correlation Heatmap:
 Identify relationships between features

# Geospatial Maps:
 Visualize visibility across latitudes and longitudes

# Example maps:

results/figures/map_all_visibility.png

results/figures/map_visibility_0.png

results/figures/map_visibility_1.png

## Future Work 

- Test additional models (XGBoost, LightGBM) for improved performance

- Explore temporal patterns (seasonal or lunar cycle effects)

- Deploy as an interactive web dashboard for public use

- Automate hyperparameter tuning and cross-validation

## License 

This project is licensed under the MIT License – see LICENSE for details.
