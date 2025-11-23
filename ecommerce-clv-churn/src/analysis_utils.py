"""
Reusable analysis utility functions for E-commerce analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


# ============================================================================
# PRODUCT ANALYSIS FUNCTIONS
# ============================================================================

def get_product_details(df, stock_code):
    """
    Display detailed information about a specific product in formatted output only.

    Args:
        df (pd.DataFrame): Transactions dataframe
        stock_code (str): Product stock code
    """
    product_data = df[df['stock_code'] == stock_code]
    
    if len(product_data) == 0:
        print(f"❌ Product '{stock_code}' not found!")
        return

    total_revenue = product_data['total_price'].sum()
    first_sale = product_data['invoice_date'].min().date()
    last_sale = product_data['invoice_date'].max().date()

    print("=" * 80)
    print(f"📊 PRODUCT ANALYSIS: {stock_code}")
    print("=" * 80)
    print(f"\nDescription: {product_data['description'].iloc[0]}")
    print(f"Total Orders: {product_data['invoice_no'].nunique():,}")
    print(f"Unique Customers: {product_data['customer_id'].nunique():,}")
    print(f"Total Quantity Sold: {product_data['quantity'].sum():,.0f}")
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Average Price: ${product_data['unit_price'].mean():.2f}")
    print(f"Average Quantity per Order: {product_data['quantity'].mean():.1f}")
    print(f"First Sale: {first_sale}")
    print(f"Last Sale:  {last_sale}")

    # Top countries by revenue
    top_countries = (
        product_data.groupby('country')['total_price']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    print(f"\nTop 5 Countries:")
    for country, revenue in top_countries.items():
        pct = (revenue / total_revenue) * 100 if total_revenue > 0 else 0
        print(f"  • {country}: ${revenue:,.2f} ({pct:.1f}%)")

    print("=" * 80)



def get_top_products(df, by='orders', top_n=20):
    """
    Get top N products by different metrics
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        by (str): Metric to sort by ('orders', 'quantity', 'revenue', 'customers')
        top_n (int): Number of top products to return
        
    Returns:
        pd.DataFrame: Top products dataframe
    """
    product_stats = df.groupby(['stock_code', 'description']).agg({
        'invoice_no': 'nunique',
        'customer_id': 'nunique',
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index()
    
    product_stats.columns = ['stock_code', 'description', 'num_orders', 
                             'num_customers', 'total_quantity', 'total_revenue']
    
    sort_mapping = {
        'orders': 'num_orders',
        'quantity': 'total_quantity',
        'revenue': 'total_revenue',
        'customers': 'num_customers'
    }
    
    sort_col = sort_mapping.get(by, 'num_orders')
    return product_stats.sort_values(sort_col, ascending=False).head(top_n)


def analyze_product_trend(df, stock_code, period='M'):
    """
    Analyze sales trend for a specific product over time
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        stock_code (str): Product stock code
        period (str): Time period ('D', 'W', 'M' for day, week, month)
        
    Returns:
        pd.DataFrame: Time series of product sales
    """
    product_data = df[df['stock_code'] == stock_code].copy()
    
    if len(product_data) == 0:
        print(f"Product {stock_code} not found!")
        return None
    
    product_data['period'] = pd.to_datetime(product_data['invoice_date']).dt.to_period(period)
    
    trend = product_data.groupby('period').agg({
        'quantity': 'sum',
        'total_price': 'sum',
        'invoice_no': 'nunique',
        'customer_id': 'nunique'
    }).reset_index()
    
    trend.columns = ['period', 'quantity', 'revenue', 'orders', 'customers']
    trend['period'] = trend['period'].astype(str)
    
    return trend


# ============================================================================
# CUSTOMER ANALYSIS FUNCTIONS
# ============================================================================

def get_customer_profile(df, customer_id):
    """
    Get detailed profile of a specific customer
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        customer_id (int): Customer ID
        
    Returns:
        dict: Customer profile details
    """
    customer_data = df[df['customer_id'] == customer_id]
    
    if len(customer_data) == 0:
        print(f"Customer {customer_id} not found!")
        return None
    
    profile = {
        'customer_id': customer_id,
        'country': customer_data['country'].iloc[0],
        'total_orders': customer_data['invoice_no'].nunique(),
        'total_items': customer_data['quantity'].sum(),
        'total_revenue': customer_data['total_price'].sum(),
        'avg_order_value': customer_data.groupby('invoice_no')['total_price'].sum().mean(),
        'first_purchase': customer_data['invoice_date'].min(),
        'last_purchase': customer_data['invoice_date'].max(),
        'unique_products': customer_data['stock_code'].nunique()
    }
    
    profile['customer_lifespan_days'] = (profile['last_purchase'] - profile['first_purchase']).days
    profile['avg_days_between_orders'] = profile['customer_lifespan_days'] / profile['total_orders'] if profile['total_orders'] > 1 else 0
    
    # Print formatted output
    print("="*80)
    print(f"👤 CUSTOMER PROFILE: {customer_id}")
    print("="*80)
    print(f"\nCountry: {profile['country']}")
    print(f"Total Orders: {profile['total_orders']:,}")
    print(f"Total Items Purchased: {profile['total_items']:,.0f}")
    print(f"Total Revenue: ${profile['total_revenue']:,.2f}")
    print(f"Average Order Value: ${profile['avg_order_value']:.2f}")
    print(f"Unique Products Purchased: {profile['unique_products']:,}")
    print(f"First Purchase: {profile['first_purchase'].date()}")
    print(f"Last Purchase: {profile['last_purchase'].date()}")
    print(f"Customer Lifespan: {profile['customer_lifespan_days']} days")
    
    if profile['total_orders'] > 1:
        print(f"Avg Days Between Orders: {profile['avg_days_between_orders']:.1f}")
    
    # Top purchased products
    top_products = customer_data.groupby(['stock_code', 'description'])['quantity'].sum().sort_values(ascending=False).head(5)
    print(f"\nTop 5 Most Purchased Products:")
    for (code, desc), qty in top_products.items():
        print(f"  • {code}: {desc[:40]} (Qty: {qty:.0f})")
    
    return profile


def get_customer_summary(df):
    """
    Get aggregated customer-level summary
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        
    Returns:
        pd.DataFrame: Customer summary dataframe
    """
    customer_summary = df.groupby('customer_id').agg({
        'invoice_no': 'nunique',
        'total_price': ['sum', 'mean'],
        'invoice_date': ['min', 'max'],
        'stock_code': 'nunique',
        'country': 'first'
    }).reset_index()
    
    customer_summary.columns = ['customer_id', 'num_orders', 'total_revenue', 
                                 'avg_order_value', 'first_purchase', 'last_purchase',
                                 'unique_products', 'country']
    
    customer_summary['customer_lifespan_days'] = (
        customer_summary['last_purchase'] - customer_summary['first_purchase']
    ).dt.days
    
    return customer_summary


def segment_customers_simple(customer_summary):
    """
    Create simple customer segments based on purchase behavior
    
    Args:
        customer_summary (pd.DataFrame): Customer summary from get_customer_summary()
        
    Returns:
        pd.DataFrame: Customer summary with segment labels
    """
    customer_summary = customer_summary.copy()
    
    # Define segments
    def assign_segment(row):
        if row['num_orders'] >= 10 and row['total_revenue'] >= 1000:
            return 'VIP'
        elif row['num_orders'] >= 5 and row['total_revenue'] >= 500:
            return 'Loyal'
        elif row['num_orders'] >= 3:
            return 'Regular'
        elif row['num_orders'] == 1:
            return 'One-Time'
        else:
            return 'Occasional'
    
    customer_summary['segment'] = customer_summary.apply(assign_segment, axis=1)
    
    return customer_summary


# ============================================================================
# TIME SERIES ANALYSIS FUNCTIONS
# ============================================================================

def get_monthly_metrics(df):
    """
    Calculate monthly business metrics
    
    Args:
        df (pd.DataFrame): Transactions dataframe with invoice_date column
        
    Returns:
        pd.DataFrame: Monthly metrics dataframe
    """
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy['invoice_date']).dt.to_period('M')
    
    monthly = df_copy.groupby('month').agg({
        'total_price': 'sum',
        'customer_id': 'nunique',
        'invoice_no': 'nunique',
        'stock_code': 'nunique'
    }).reset_index()
    
    monthly.columns = ['month', 'revenue', 'unique_customers', 'orders', 'unique_products']
    monthly['month'] = monthly['month'].astype(str)
    monthly['avg_order_value'] = monthly['revenue'] / monthly['orders']
    monthly['revenue_per_customer'] = monthly['revenue'] / monthly['unique_customers']
    
    return monthly


def calculate_growth_rates(monthly_df, metric_col='revenue'):
    """
    Calculate month-over-month and year-over-year growth rates
    
    Args:
        monthly_df (pd.DataFrame): Monthly metrics dataframe
        metric_col (str): Column to calculate growth for
        
    Returns:
        pd.DataFrame: Monthly data with growth rates
    """
    df = monthly_df.copy()
    
    # Month-over-month growth
    df['mom_growth'] = df[metric_col].pct_change() * 100
    
    # Year-over-year growth (if applicable)
    if len(df) >= 12:
        df['yoy_growth'] = df[metric_col].pct_change(periods=12) * 100
    
    return df


# ============================================================================
# COHORT ANALYSIS FUNCTIONS
# ============================================================================

def create_cohort_analysis(df):
    """
    Create cohort retention analysis based on first purchase month
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        
    Returns:
        pd.DataFrame: Cohort retention matrix
    """
    df_copy = df.copy()
    df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
    df_copy['order_month'] = df_copy['invoice_date'].dt.to_period('M')
    
    # Get first purchase month for each customer
    df_copy['cohort_month'] = df_copy.groupby('customer_id')['invoice_date'].transform('min').dt.to_period('M')
    
    # Calculate cohort index (months since first purchase)
    def get_cohort_period(row):
        cohort_index = (row['order_month'].to_timestamp() - row['cohort_month'].to_timestamp())
        return cohort_index.days // 30
    
    df_copy['cohort_index'] = df_copy.apply(get_cohort_period, axis=1)
    
    # Create cohort matrix
    cohort_data = df_copy.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
    cohort_data.rename(columns={'customer_id': 'unique_customers'}, inplace=True)
    
    # Pivot to create cohort matrix
    cohort_matrix = cohort_data.pivot_table(
        index='cohort_month',
        columns='cohort_index',
        values='unique_customers'
    )
    
    return cohort_matrix


# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def plot_top_items(df, by='products', top_n=10, figsize=(12, 6)):
    """
    Plot horizontal bar chart of top items
    
    Args:
        df (pd.DataFrame): Dataframe with items and values
        by (str): What to plot ('products', 'customers', 'countries')
        top_n (int): Number of items to display
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    if by == 'products':
        top_data = get_top_products(df, by='revenue', top_n=top_n)
        plt.barh(range(len(top_data)), top_data['total_revenue'], color='#2E86AB', alpha=0.8)
        plt.yticks(range(len(top_data)), [desc[:40] for desc in top_data['description']])
        plt.xlabel('Revenue ($)')
        plt.title(f'Top {top_n} Products by Revenue', fontsize=14, fontweight='bold')
    
    elif by == 'customers':
        customer_rev = df.groupby('customer_id')['total_price'].sum().sort_values(ascending=False).head(top_n)
        plt.barh(range(len(customer_rev)), customer_rev.values, color='#A23B72', alpha=0.8)
        plt.yticks(range(len(customer_rev)), [f'Customer {cid}' for cid in customer_rev.index])
        plt.xlabel('Revenue ($)')
        plt.title(f'Top {top_n} Customers by Revenue', fontsize=14, fontweight='bold')
    
    elif by == 'countries':
        country_rev = df.groupby('country')['total_price'].sum().sort_values(ascending=False).head(top_n)
        plt.barh(range(len(country_rev)), country_rev.values, color='#F18F01', alpha=0.8)
        plt.yticks(range(len(country_rev)), country_rev.index)
        plt.xlabel('Revenue ($)')
        plt.title(f'Top {top_n} Countries by Revenue', fontsize=14, fontweight='bold')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_time_series(df, metric='revenue', period='M', figsize=(14, 6)):
    """
    Plot time series of a specific metric
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        metric (str): Metric to plot ('revenue', 'customers', 'orders')
        period (str): Time period ('D', 'W', 'M')
        figsize (tuple): Figure size
    """
    df_copy = df.copy()
    df_copy['period'] = pd.to_datetime(df_copy['invoice_date']).dt.to_period(period)
    
    if metric == 'revenue':
        data = df_copy.groupby('period')['total_price'].sum()
        ylabel = 'Revenue ($)'
        title = f'Revenue Over Time ({period})'
    elif metric == 'customers':
        data = df_copy.groupby('period')['customer_id'].nunique()
        ylabel = 'Unique Customers'
        title = f'Unique Customers Over Time ({period})'
    elif metric == 'orders':
        data = df_copy.groupby('period')['invoice_no'].nunique()
        ylabel = 'Number of Orders'
        title = f'Orders Over Time ({period})'
    
    plt.figure(figsize=figsize)
    plt.plot(data.index.astype(str), data.values, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Period')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_summary_report(df, output_path='../reports/summary_report.csv'):
    """
    Export comprehensive summary report
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        output_path (str): Output file path
    """
    customer_summary = get_customer_summary(df)
    customer_summary = segment_customers_simple(customer_summary)
    
    customer_summary.to_csv(output_path, index=False)
    print(f"✅ Summary report exported to {output_path}")


def print_business_summary(df):
    """
    Print comprehensive business summary
    
    Args:
        df (pd.DataFrame): Transactions dataframe
    """
    print("="*80)
    print("📊 BUSINESS SUMMARY")
    print("="*80)
    
    print(f"\n📅 Time Period:")
    print(f"   From: {df['invoice_date'].min().date()}")
    print(f"   To: {df['invoice_date'].max().date()}")
    print(f"   Duration: {(df['invoice_date'].max() - df['invoice_date'].min()).days} days")
    
    print(f"\n💰 Revenue Metrics:")
    print(f"   Total Revenue: ${df['total_price'].sum():,.2f}")
    print(f"   Avg Transaction Value: ${df['total_price'].mean():.2f}")
    print(f"   Median Transaction Value: ${df['total_price'].median():.2f}")
    
    print(f"\n👥 Customer Metrics:")
    print(f"   Total Customers: {df['customer_id'].nunique():,}")
    print(f"   Avg Revenue per Customer: ${df.groupby('customer_id')['total_price'].sum().mean():,.2f}")
    print(f"   Avg Orders per Customer: {df.groupby('customer_id')['invoice_no'].nunique().mean():.2f}")
    
    print(f"\n📦 Order Metrics:")
    print(f"   Total Orders: {df['invoice_no'].nunique():,}")
    print(f"   Total Items Sold: {df['quantity'].sum():,.0f}")
    print(f"   Unique Products: {df['stock_code'].nunique():,}")
    
    print(f"\n🌍 Geographic Coverage:")
    print(f"   Countries Served: {df['country'].nunique()}")
    print(f"   Top Country: {df.groupby('country')['total_price'].sum().idxmax()}")
    
    print("\n" + "="*80)