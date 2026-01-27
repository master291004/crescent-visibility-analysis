from data_processing import load_raw_data,clean_data,feature_engineering, save_cleaned_data
from models import train_random_forest, train_logistic_regression,  predict, predict_proba
from evaluation import plot_confusion_matrix,  plot_roc_curve, plot_roc_curve_comparison, classification_report_text
from visualization import distribution_visibility, plot_histograms, plot_correlation_heatmap, plot_visibility_map
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# 1. Load and preprocess data
# ---------------------------

raw_file = 'C:\\Users\\miled\\OneDrive\\Documents\\GitHub\\crescent-visibility-analysis\\data\\raw\\Final.csv'
processed_file = 'C:\\Users\\miled\\OneDrive\\Documents\\GitHub\\crescent-visibility-analysis\\data\\processed\\cleaned_data.csv'

df = load_raw_data(raw_file)
df = clean_data(df)
df = feature_engineering(df)
save_cleaned_data(df, processed_file)

print("Data loaded and preprocessed successfully.")

#---------------------------
# 2.Geospatial Visualizations 
#---------------------------
import geopandas as gpd 
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
    crs="EPSG:4326"
)

# Create visibility label for plotting
gdf["Visibility_label"] = gdf["Visibility"].map({0: "Invisible", 1: "Visible"})

# Plot all points
plot_visibility_map(
    gdf,
    title="Moon Visibility Observations (0 & 1)",
    filename="results/figures/map_all_visibility.png",
    colors=["skyblue", "orange"]
)

# Plot only zeros
plot_visibility_map(
    gdf[gdf["Visibility"]==0],
    title="Moon Visibility Observations (0 only)",
    filename="results/figures/map_visibility_0.png",
    colors=["skyblue"]
)

# Plot only ones
plot_visibility_map(
    gdf[gdf["Visibility"]==1],
    title="Moon Visibility Observations (1 only)",
    filename="results/figures/map_visibility_1.png",
    colors=["orange"]
)

#---------------------------
# 3. Visualizations of feature distributions
#---------------------------
distribution_visibility(df, filename="results/figures/plot_distribution_Visibility.png")
plot_histograms(df, filename="results/figures/plot_histograms.png")
plot_correlation_heatmap(df, filename="results/figures/plot_correlation_heatmap.png")

#---------------------------
# 4. Split features and labels 
#---------------------------
X = df.drop(columns=["Visibility"])
y = df["Visibility"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Data split into training and testing sets.")

# ---------------------------
# 5. Train model random forest classifier
#---------------------------
randomForest=train_random_forest(X_train,y_train,100,None,42,None)

#---------------------------
# 6. Predictions and evaluations
#---------------------------
y_pred = predict(randomForest, X_test)
y_prob = predict_proba(randomForest,X_test)


# Confusion matrix
plot_confusion_matrix(y_test,y_pred,"confusion matrix",filename="results/metrics/confusion_matrix.png")


# Classification report
classification = classification_report(y_test, y_pred, output_dict=True)
print(classification_report_text(y_test, y_pred))

# ROC curve
auc = plot_roc_curve(y_test, y_prob, filename="results/metrics/roc_curve.png")
print(f"AUC: {auc:.2f}")

#---------------------------
# 7. Train Logistic Regression model
#--------------------------
logreg = train_logistic_regression(X_train,y_train)

y_pred_lr = predict(logreg, X_test)
y_prob_lr = predict_proba(logreg, X_test)

# Confusion matrix
plot_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression",
                      filename="results/metrics/confusion_matrix_lr.png")

# Classification report
classification_lr = classification_report(y_test, y_pred_lr, output_dict=True)
print("Logistic Regression Performance:")
print(classification_report_text(y_test, y_pred_lr))

# ROC curve
auc_lr = plot_roc_curve(y_test, y_prob_lr, filename="results/metrics/roc_curve_lr.png")
print(f"Logistic Regression AUC: {auc_lr:.2f}")


#---------------------------
# 8. Model Comparison
#---------------------------

print(f"{'Model':<20}{'Accuracy':<10}{'F1-score':<10}{'AUC':<10}")
print(f"{'Random Forest':<20}{classification['accuracy']:<10.2f}{classification['weighted avg']['f1-score']:<10.2f}{auc:<10.2f}")
print(f"{'Logistic Regression':<20}{classification_lr['accuracy']:<10.2f}{classification_lr['weighted avg']['f1-score']:<10.2f}{auc_lr:<10.2f}")

plot_roc_curve_comparison(
    y_test,
    {
        "Random Forest": y_prob,
        "Logistic Regression": y_prob_lr
    },
    filename="results/metrics/roc_comparison.png"
)


print("Pipeline completed successfully!")


 