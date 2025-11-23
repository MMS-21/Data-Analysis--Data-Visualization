"""
Customer Lifetime Value (CLV) and RFM Analysis Functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# RFM ANALYSIS FUNCTIONS
# ============================================================================

def calculate_rfm(df, reference_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
    
    Args:
        df (pd.DataFrame): Transactions dataframe with columns:
                          ['customer_id', 'invoice_date', 'invoice_no', 'total_price']
        reference_date (datetime): Date to calculate recency from (default: max date + 1 day)
        
    Returns:
        pd.DataFrame: RFM metrics for each customer
    """
    # Set reference date (typically day after last transaction)
    if reference_date is None:
        reference_date = df['invoice_date'].max() + timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (reference_date - x.max()).days,  # Recency
        'invoice_no': 'nunique',                                     # Frequency
        'total_price': 'sum'                                         # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    return rfm


def calculate_rfm_scores(rfm_df, quantiles=5):
    """
    Calculate RFM scores using quintile-based scoring (1-5)
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe from calculate_rfm()
        quantiles (int): Number of quantiles (default: 5 for 1-5 scoring)
        
    Returns:
        pd.DataFrame: RFM dataframe with scores added
    """
    rfm_scored = rfm_df.copy()
    
    # Calculate quintiles (lower recency is better, higher F and M are better)
    rfm_scored['r_score'] = pd.qcut(rfm_scored['recency'], q=quantiles, labels=range(quantiles, 0, -1), duplicates='drop')
    rfm_scored['f_score'] = pd.qcut(rfm_scored['frequency'].rank(method='first'), q=quantiles, labels=range(1, quantiles+1), duplicates='drop')
    rfm_scored['m_score'] = pd.qcut(rfm_scored['monetary'].rank(method='first'), q=quantiles, labels=range(1, quantiles+1), duplicates='drop')
    
    # Convert to integers
    rfm_scored['r_score'] = rfm_scored['r_score'].astype(int)
    rfm_scored['f_score'] = rfm_scored['f_score'].astype(int)
    rfm_scored['m_score'] = rfm_scored['m_score'].astype(int)
    
    # Calculate combined RFM score
    rfm_scored['rfm_score'] = (rfm_scored['r_score'].astype(str) + 
                                rfm_scored['f_score'].astype(str) + 
                                rfm_scored['m_score'].astype(str))
    
    # Calculate total score (sum of R, F, M)
    rfm_scored['total_score'] = rfm_scored['r_score'] + rfm_scored['f_score'] + rfm_scored['m_score']
    
    return rfm_scored


def segment_customers_rfm(rfm_scored_df):
    """
    Segment customers based on RFM scores into business-meaningful segments
    
    Args:
        rfm_scored_df (pd.DataFrame): RFM dataframe with scores
        
    Returns:
        pd.DataFrame: RFM dataframe with segment labels
    """
    rfm_segmented = rfm_scored_df.copy()
    
    def assign_segment(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        # Champions: High R, F, M
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: High F, moderate to high R and M
        elif f >= 4 and r >= 3:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with moderate F and M
        elif r >= 4 and f >= 2 and f <= 3:
            return 'Potential Loyalists'
        
        # Recent Customers: High recency but low F and M
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        
        # Promising: Moderate recency, frequency, and monetary
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Promising'
        
        # Need Attention: Below average but not lost yet
        elif r >= 2 and r <= 3:
            return 'Need Attention'
        
        # About to Sleep: Low recency, moderate to high F and M
        elif r <= 2 and f >= 2:
            return 'About to Sleep'
        
        # At Risk: High F and M but low recency
        elif r <= 2 and f >= 4 and m >= 4:
            return 'At Risk'
        
        # Cannot Lose Them: Very low recency but high F and M
        elif r <= 1 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'
        
        # Hibernating: Low recency, low F and M
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        
        # Lost: Very low scores across the board
        else:
            return 'Lost'
    
    rfm_segmented['segment'] = rfm_segmented.apply(assign_segment, axis=1)
    
    return rfm_segmented


def get_segment_summary(rfm_segmented_df):
    """
    Get summary statistics for each customer segment
    
    Args:
        rfm_segmented_df (pd.DataFrame): RFM dataframe with segments
        
    Returns:
        pd.DataFrame: Segment summary statistics
    """
    segment_summary = rfm_segmented_df.groupby('segment').agg({
        'customer_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'sum']
    }).reset_index()
    
    segment_summary.columns = ['segment', 'num_customers', 'avg_recency', 
                                'avg_frequency', 'avg_monetary', 'total_revenue']
    
    # Calculate percentages
    segment_summary['customer_pct'] = (segment_summary['num_customers'] / 
                                        segment_summary['num_customers'].sum() * 100)
    segment_summary['revenue_pct'] = (segment_summary['total_revenue'] / 
                                       segment_summary['total_revenue'].sum() * 100)
    
    # Sort by revenue
    segment_summary = segment_summary.sort_values('total_revenue', ascending=False)
    
    return segment_summary


# ============================================================================
# HISTORIC CLV FUNCTIONS
# ============================================================================

def calculate_historic_clv(df):
    """
    Calculate historic Customer Lifetime Value (total past revenue per customer)
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        
    Returns:
        pd.DataFrame: Customer CLV dataframe
    """
    clv = df.groupby('customer_id').agg({
        'total_price': 'sum',
        'invoice_no': 'nunique',
        'invoice_date': ['min', 'max']
    }).reset_index()
    
    clv.columns = ['customer_id', 'historic_clv', 'total_orders', 
                   'first_purchase', 'last_purchase']
    
    clv['customer_lifespan_days'] = (clv['last_purchase'] - clv['first_purchase']).dt.days
    clv['avg_order_value'] = clv['historic_clv'] / clv['total_orders']
    
    return clv


# ============================================================================
# PREDICTIVE CLV FUNCTIONS (BG/NBD + Gamma-Gamma)
# ============================================================================

def prepare_clv_data(df, observation_period_end=None, prediction_period_months=12):
    """
    Prepare data for predictive CLV modeling using lifetimes library format
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        observation_period_end (datetime): End of observation period
        prediction_period_months (int): Months to predict forward
        
    Returns:
        pd.DataFrame: Customer summary for CLV modeling
    """
    if observation_period_end is None:
        observation_period_end = df['invoice_date'].max()
    
    # Calculate customer metrics
    customer_data = df.groupby('customer_id').agg({
        'invoice_date': ['min', 'max'],
        'invoice_no': 'nunique',
        'total_price': 'sum'
    }).reset_index()
    
    customer_data.columns = ['customer_id', 'first_purchase', 'last_purchase', 
                             'frequency', 'monetary_value']
    
    # Calculate recency (time between first and last purchase in days)
    customer_data['recency_days'] = (customer_data['last_purchase'] - 
                                      customer_data['first_purchase']).dt.days
    
    # Calculate T (age of customer in days from first purchase to observation end)
    customer_data['T_days'] = (observation_period_end - 
                                customer_data['first_purchase']).dt.days
    
    # Frequency is number of repeat purchases (subtract 1 for first purchase)
    customer_data['frequency'] = customer_data['frequency'] - 1
    
    # Average order value (monetary)
    customer_data['monetary_value'] = customer_data['monetary_value'] / (customer_data['frequency'] + 1)
    
    # Filter customers with at least one repeat purchase for monetary model
    customer_data_filtered = customer_data[customer_data['frequency'] > 0].copy()
    
    return customer_data_filtered


def calculate_simple_predictive_clv(customer_data, prediction_months=12, discount_rate=0.01):
    """
    Calculate simple predictive CLV using historical patterns
    (Alternative to lifetimes library - no external dependencies)
    
    Args:
        customer_data (pd.DataFrame): Customer data from prepare_clv_data()
        prediction_months (int): Number of months to predict
        discount_rate (float): Monthly discount rate
        
    Returns:
        pd.DataFrame: Customer data with predicted CLV
    """
    clv_data = customer_data.copy()
    
    # Calculate purchase rate (purchases per day)
    clv_data['purchase_rate'] = clv_data['frequency'] / clv_data['T_days']
    clv_data['purchase_rate'] = clv_data['purchase_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Estimate purchases in next period
    prediction_days = prediction_months * 30
    clv_data['predicted_purchases'] = clv_data['purchase_rate'] * prediction_days
    
    # Predicted revenue (purchases * avg order value)
    clv_data['predicted_revenue'] = clv_data['predicted_purchases'] * clv_data['monetary_value']
    
    # Apply discount rate
    months = np.arange(1, prediction_months + 1)
    discount_factors = [(1 / (1 + discount_rate) ** m) for m in months]
    avg_discount_factor = np.mean(discount_factors)
    
    clv_data['predicted_clv'] = clv_data['predicted_revenue'] * avg_discount_factor
    
    return clv_data


def calculate_clv_segments(clv_data):
    """
    Segment customers by predicted CLV value
    
    Args:
        clv_data (pd.DataFrame): CLV data with predicted_clv column
        
    Returns:
        pd.DataFrame: CLV data with value segments
    """
    clv_segmented = clv_data.copy()
    
    # Calculate percentiles
    p25 = clv_segmented['predicted_clv'].quantile(0.25)
    p50 = clv_segmented['predicted_clv'].quantile(0.50)
    p75 = clv_segmented['predicted_clv'].quantile(0.75)
    p90 = clv_segmented['predicted_clv'].quantile(0.90)
    
    def assign_value_segment(clv):
        if clv >= p90:
            return 'High Value'
        elif clv >= p75:
            return 'Medium-High Value'
        elif clv >= p50:
            return 'Medium Value'
        elif clv >= p25:
            return 'Low-Medium Value'
        else:
            return 'Low Value'
    
    clv_segmented['value_segment'] = clv_segmented['predicted_clv'].apply(assign_value_segment)
    
    return clv_segmented


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_rfm_distribution(rfm_df, figsize=(15, 5)):
    """
    Plot distribution of RFM metrics
    
    Args:
        rfm_df (pd.DataFrame): RFM dataframe
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Recency
    axes[0].hist(rfm_df['recency'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_title('Recency Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Days Since Last Purchase')
    axes[0].set_ylabel('Number of Customers')
    axes[0].axvline(rfm_df['recency'].median(), color='red', linestyle='--', 
                    label=f"Median: {rfm_df['recency'].median():.0f} days")
    axes[0].legend()
    
    # Frequency
    axes[1].hist(rfm_df['frequency'], bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].set_title('Frequency Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Orders')
    axes[1].set_ylabel('Number of Customers')
    axes[1].axvline(rfm_df['frequency'].median(), color='red', linestyle='--',
                    label=f"Median: {rfm_df['frequency'].median():.0f} orders")
    axes[1].legend()
    
    # Monetary
    axes[2].hist(rfm_df['monetary'], bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[2].set_title('Monetary Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Total Spending ($)')
    axes[2].set_ylabel('Number of Customers')
    axes[2].axvline(rfm_df['monetary'].median(), color='red', linestyle='--',
                    label=f"Median: ${rfm_df['monetary'].median():.2f}")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


def plot_rfm_segments(rfm_segmented_df, figsize=(14, 8)):
    """
    Visualize RFM customer segments
    
    Args:
        rfm_segmented_df (pd.DataFrame): RFM dataframe with segments
        figsize (tuple): Figure size
    """
    segment_summary = get_segment_summary(rfm_segmented_df)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Customer count by segment
    axes[0, 0].barh(segment_summary['segment'], segment_summary['num_customers'], 
                    color='#2E86AB', alpha=0.8)
    axes[0, 0].set_title('Number of Customers by Segment', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Customers')
    axes[0, 0].invert_yaxis()
    
    # 2. Revenue by segment
    axes[0, 1].barh(segment_summary['segment'], segment_summary['total_revenue'], 
                    color='#A23B72', alpha=0.8)
    axes[0, 1].set_title('Total Revenue by Segment', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Revenue ($)')
    axes[0, 1].invert_yaxis()
    axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    # 3. Customer percentage
    colors = plt.cm.Set3(range(len(segment_summary)))
    axes[1, 0].pie(segment_summary['customer_pct'], labels=segment_summary['segment'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title('Customer Distribution', fontsize=12, fontweight='bold')
    
    # 4. Revenue percentage
    axes[1, 1].pie(segment_summary['revenue_pct'], labels=segment_summary['segment'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Revenue Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_rfm_heatmap(rfm_scored_df, figsize=(10, 8)):
    """
    Create heatmap showing relationship between RFM scores
    
    Args:
        rfm_scored_df (pd.DataFrame): RFM dataframe with scores
        figsize (tuple): Figure size
    """
    # Create pivot table for heatmap (R vs F, with M as values)
    pivot_data = rfm_scored_df.groupby(['r_score', 'f_score']).agg({
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    pivot_monetary = pivot_data.pivot(index='r_score', columns='f_score', values='monetary')
    pivot_count = pivot_data.pivot(index='r_score', columns='f_score', values='customer_id')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Monetary heatmap
    sns.heatmap(pivot_monetary, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Avg Monetary Value ($)'})
    axes[0].set_title('Average Monetary Value by R & F Scores', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency Score')
    axes[0].set_ylabel('Recency Score')
    
    # Customer count heatmap
    sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='Blues', 
                ax=axes[1], cbar_kws={'label': 'Number of Customers'})
    axes[1].set_title('Customer Count by R & F Scores', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency Score')
    axes[1].set_ylabel('Recency Score')
    
    plt.tight_layout()
    plt.show()


def plot_clv_distribution(clv_data, figsize=(14, 5)):
    """
    Plot CLV distribution and segments
    
    Args:
        clv_data (pd.DataFrame): CLV data with predicted_clv
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # CLV distribution
    axes[0].hist(clv_data['predicted_clv'], bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_title('Predicted CLV Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted CLV ($)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].axvline(clv_data['predicted_clv'].median(), color='red', linestyle='--',
                    label=f"Median: ${clv_data['predicted_clv'].median():.2f}")
    axes[0].legend()
    
    # CLV distribution (log scale for better visibility)
    axes[1].hist(np.log10(clv_data['predicted_clv'] + 1), bins=50, 
                 color='#A23B72', alpha=0.7, edgecolor='black')
    axes[1].set_title('Predicted CLV Distribution (Log Scale)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Log10(Predicted CLV)')
    axes[1].set_ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.show()