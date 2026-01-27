import pandas as pd 


def load_raw_data(filepath):
    """
    Load raw data from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file

    Returns
    -------
    df : pandas.DataFrame
        Loaded data
    """
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Clean and preprocess the raw data.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data

    Returns
    -------
    df : pandas.DataFrame
        Cleaned data
    """
    cols = ["Lat", "Long", "ArcV", "ArcL", "DAZ", "W", "MA" , "JD", "Ele", "Moonset", "Sunset", "V"]
    df = df[cols]
    df = df.dropna()
    return df
def feature_engineering(df):
    """
    Perform feature engineering on the cleaned data.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned data

    Returns
    -------
    df : pandas.DataFrame
        Data with engineered features
    """
    #Convert Sunset and Moonset to datetime
    df["Sunset"] = pd.to_datetime(df["Sunset"], format="%H:%M", errors="coerce")
    df["Moonset"] = pd.to_datetime(df["Moonset"], format="%H:%M", errors="coerce")
    
    #Add a new column
    df["Lag"] = df["Moonset"] - df["Sunset"]
    df["Lag_minutes"] = df["Lag"].dt.total_seconds() / 60
    
    #Drop unecessary columns
    df = df.drop(columns=["Lag", "Moonset", "Sunset"])
    
    #Encode Visibility 
    df["V"] = df["V"].map({"V": 1, "I": 0})
    
    #Rename columns 
    df = df.rename(columns={
        "W": "Crescent Width",
        "V": "Visibility"
    })
    return df

def save_cleaned_data(df, filepath):
    """
    Save the processed dataframe to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed dataframe
    filepath : str
        Path to save the CSV file
    """
    df.to_csv(filepath, index=False)






