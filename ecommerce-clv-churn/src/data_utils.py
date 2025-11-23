"""
Data cleaning utilities for Online Retail dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_raw_data(filepath):
    """
    Load raw Online Retail data from CSV/Excel
    
    Args:
        filepath (str): Path to raw data file
        
    Returns:
        pd.DataFrame: Raw dataframe with all sheets combined
    """
    if filepath.endswith('.xlsx'):
        # Read all sheets into a dictionary
        all_sheets = pd.read_excel(filepath, sheet_name=None)
        
        # Combine all sheets
        df = pd.concat(all_sheets.values(), ignore_index=True)
        
    else:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_retail_data(df):
    """
    Clean Online Retail dataset with comprehensive preprocessing
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n=== Starting Data Cleaning ===")
    initial_rows = len(df)
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Remove rows with missing CustomerID
    print(f"Missing CustomerID: {df_clean['CustomerID'].isna().sum()}")
    df_clean = df_clean[df_clean['CustomerID'].notna()]
    print(f"After removing missing CustomerID: {len(df_clean)} rows")
    
    # 2. Remove cancelled orders (InvoiceNo starting with 'C')
    cancelled = df_clean['InvoiceNo'].astype(str).str.startswith('C')
    print(f"Cancelled orders: {cancelled.sum()}")
    df_clean = df_clean[~cancelled]
    print(f"After removing cancellations: {len(df_clean)} rows")
    
    # 3. Remove negative/zero quantities and prices
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    print(f"After removing negative/zero values: {len(df_clean)} rows")
    
    # 4. Handle missing descriptions
    print(f"Missing Description: {df_clean['Description'].isna().sum()}")
    df_clean = df_clean[df_clean['Description'].notna()]
    
    # 5. Convert data types
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str)
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Description'] = df_clean['Description'].str.strip()
    df_clean['Country'] = df_clean['Country'].str.strip()
    
    # 6. Create derived features
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    
    # 7. Remove outliers (optional - using IQR method on TotalPrice)
    Q1 = df_clean['TotalPrice'].quantile(0.25)
    Q3 = df_clean['TotalPrice'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Using 3*IQR for less aggressive filtering
    upper_bound = Q3 + 3 * IQR
    
    outliers = (df_clean['TotalPrice'] < lower_bound) | (df_clean['TotalPrice'] > upper_bound)
    print(f"Outliers detected: {outliers.sum()}")
    df_clean = df_clean[~outliers]
    
    print(f"\n=== Cleaning Complete ===")
    print(f"Rows removed: {initial_rows - len(df_clean)} ({(initial_rows - len(df_clean))/initial_rows*100:.2f}%)")
    print(f"Final dataset: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    print(f"Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")
    print(f"Unique customers: {df_clean['CustomerID'].nunique()}")
    print(f"Unique products: {df_clean['StockCode'].nunique()}")
    
    return df_clean


def generate_summary_stats(df):
    """
    Generate summary statistics for cleaned data
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_transactions': len(df),
        'unique_customers': df['CustomerID'].nunique(),
        'unique_products': df['StockCode'].nunique(),
        'unique_invoices': df['InvoiceNo'].nunique(),
        'countries': df['Country'].nunique(),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'total_revenue': df['TotalPrice'].sum(),
        'avg_order_value': df['TotalPrice'].mean(),
        'median_order_value': df['TotalPrice'].median(),
    }
    
    return summary


def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Output file path
    """
    # Convert Period to string for CSV compatibility
    df_save = df.copy()
    df_save['YearMonth'] = df_save['YearMonth'].astype(str)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_save.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")


def main():
    """
    Main pipeline execution
    """

    # Define paths
    cwd = os.getcwd()
    pcwd = os.path.dirname(cwd)
    RAW_DATA_PATH = cwd + '\\data\\raw\\online_retail_II.xlsx'  # Adjust filename as needed
    CLEANED_DATA_PATH = cwd + '\\data\\cleaned\\retail_clean.xlsx'
    
    # Execute pipeline
    print("Loading raw data...")
    df_raw = load_raw_data(RAW_DATA_PATH)
    
    print("\nCleaning data...")
    df_clean = clean_retail_data(df_raw)
    
    print("\nGenerating summary statistics...")
    summary = generate_summary_stats(df_clean)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nSaving cleaned data...")
    save_cleaned_data(df_clean, CLEANED_DATA_PATH)
    
    print("\n✅ Data cleaning pipeline complete!")


if __name__ == "__main__":
    main()