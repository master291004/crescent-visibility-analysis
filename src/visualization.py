import geopandas as gpd 
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.colors import ListedColormap    
import numpy as np
import seaborn as sns
import geopandas as gpd

df=pd.read_csv("data/processed/cleaned_data.csv")

# --- 1.distribution of visibility ---
def distribution_visibility(df, filename):
    """
    Plot and save the distribution of the visibility column 
    
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the 'Visibility' column 
    """
    plt.figure(figsize=(6,4))
    sns.countplot(x='Visibility', data=df)
    plt.title("Distribution of Visibility")
    plt.savefig(filename, dpi=300)
    plt.show()


# --- 2. histograms for all numeric features ---
def plot_histograms(df,filename):
    """
    Plot and save histograms for all numeric features in the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing numeric features
    """
    df.hist(bins=20, figsize=(12, 9))
    plt.suptitle("Feature Distributions")
    plt.savefig(filename, dpi=300)
    plt.show()

# --- 3. Correlation heatmap ---
def plot_correlation_heatmap(df, filename):
    """
    Plot the correlation heatmap of the dataframe features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing features to analyze
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(filename, dpi=300)
    plt.show()



# --- 4. Scatter plot: Lat vs Long colored by Visibility ---

# --- Common function to plot map ---
def plot_visibility_map(gdf_plot, title, filename, colors):
    """
    Plot the geospatial visibility map.

    Parameters
    ----------
    gdf_plot : GeoDataframe
        Data with geometry and visibility labels
    title : str
        Title of the plot 
    filename : str
        File path to save figure
    colors : List
        List of colors for visibility categories
    """
    
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    world.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_plot.plot(
    ax=ax,
    column="Visibility_label",
    categorical=True,
    cmap=ListedColormap(colors),
    legend=True,
    markersize=30,
    alpha=0.7
)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(filename, dpi=300)
    plt.show()
