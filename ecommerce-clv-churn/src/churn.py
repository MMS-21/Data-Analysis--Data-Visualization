"""
Churn Prediction Modeling Functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CHURN DEFINITION & FEATURE ENGINEERING
# ============================================================================

def define_churn(df, churn_threshold_days=90, reference_date=None):
    """
    Define churn based on recency threshold
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        churn_threshold_days (int): Days without purchase to consider churned
        reference_date (datetime): Reference date for calculation
        
    Returns:
        pd.DataFrame: Customer data with churn label
    """
    if reference_date is None:
        reference_date = df['invoice_date'].max()
    
    # Calculate last purchase date for each customer
    customer_last_purchase = df.groupby('customer_id').agg({
        'invoice_date': 'max'
    }).reset_index()
    
    customer_last_purchase.columns = ['customer_id', 'last_purchase_date']
    
    # Calculate days since last purchase
    customer_last_purchase['days_since_last_purchase'] = (
        reference_date - customer_last_purchase['last_purchase_date']
    ).dt.days
    
    # Define churn
    customer_last_purchase['is_churned'] = (
        customer_last_purchase['days_since_last_purchase'] > churn_threshold_days
    ).astype(int)
    
    return customer_last_purchase


def engineer_churn_features(df, churn_labels):
    """
    Create features for churn prediction model
    
    Args:
        df (pd.DataFrame): Transactions dataframe
        churn_labels (pd.DataFrame): Customer churn labels from define_churn()
        
    Returns:
        pd.DataFrame: Feature matrix for modeling
    """
    # RFM features
    reference_date = df['invoice_date'].max()
    
    features = df.groupby('customer_id').agg({
        'invoice_date': ['min', 'max'],
        'invoice_no': 'nunique',
        'total_price': ['sum', 'mean', 'std', 'min', 'max'],
        'quantity': ['sum', 'mean'],
        'stock_code': 'nunique',
        'country': 'first'
    }).reset_index()
    
    # Flatten column names
    features.columns = ['_'.join(col).strip('_') for col in features.columns.values]
    features.rename(columns={'customer_id_': 'customer_id'}, inplace=True)
    
    # Calculate derived features
    features['recency_days'] = (reference_date - features['invoice_date_max']).dt.days
    features['customer_age_days'] = (features['invoice_date_max'] - features['invoice_date_min']).dt.days
    features['frequency'] = features['invoice_no_nunique']
    features['monetary'] = features['total_price_sum']
    features['avg_order_value'] = features['total_price_mean']
    features['total_quantity'] = features['quantity_sum']
    features['unique_products'] = features['stock_code_nunique']
    
    # Order value consistency (lower std = more consistent)
    features['order_value_std'] = features['total_price_std'].fillna(0)
    features['order_value_cv'] = (features['order_value_std'] / features['avg_order_value']).fillna(0)
    features['order_value_cv'] = features['order_value_cv'].replace([np.inf, -np.inf], 0)
    
    # Purchase patterns
    features['avg_days_between_orders'] = features['customer_age_days'] / (features['frequency'] - 1)
    features['avg_days_between_orders'] = features['avg_days_between_orders'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Monetary features
    features['min_order_value'] = features['total_price_min']
    features['max_order_value'] = features['total_price_max']
    features['order_value_range'] = features['max_order_value'] - features['min_order_value']
    
    # Items per order
    features['avg_items_per_order'] = features['total_quantity'] / features['frequency']
    
    # Product diversity (higher = more diverse purchases)
    features['product_diversity_ratio'] = features['unique_products'] / features['frequency']
    
    # Recent activity indicator
    features['purchased_last_30_days'] = (features['recency_days'] <= 30).astype(int)
    features['purchased_last_60_days'] = (features['recency_days'] <= 60).astype(int)
    
    # Country (encode as feature - keep only top countries, others as 'Other')
    country_counts = df['country'].value_counts()
    top_countries = country_counts.head(10).index.tolist()
    features['country_clean'] = features['country_first'].apply(
        lambda x: x if x in top_countries else 'Other'
    )
    
    # Merge with churn labels
    features = features.merge(churn_labels[['customer_id', 'is_churned']], on='customer_id', how='inner')
    
    # Select relevant features for modeling
    feature_cols = [
        'recency_days', 'frequency', 'monetary', 'avg_order_value',
        'customer_age_days', 'total_quantity', 'unique_products',
        'order_value_std', 'order_value_cv', 'avg_days_between_orders',
        'min_order_value', 'max_order_value', 'order_value_range',
        'avg_items_per_order', 'product_diversity_ratio',
        'purchased_last_30_days', 'purchased_last_60_days', 'country_clean'
    ]
    
    return features[['customer_id'] + feature_cols + ['is_churned']]


def prepare_model_data(features_df):
    """
    Prepare data for machine learning (encode categoricals, split data)
    
    Args:
        features_df (pd.DataFrame): Feature dataframe from engineer_churn_features()
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # Separate features and target
    X = features_df.drop(['customer_id', 'is_churned'], axis=1)
    y = features_df['is_churned']
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=['country_clean'], drop_first=True)
    
    # Store feature names
    feature_names = X_encoded.columns.tolist()
    
    # Split data (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: model, predictions, probabilities, metrics
    """
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return model, y_pred, y_prob, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: model, predictions, probabilities, metrics
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return model, y_pred, y_prob, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train Gradient Boosting model
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: model, predictions, probabilities, metrics
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return model, y_pred, y_prob, metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models
    
    Args:
        models_dict (dict): Dictionary of {model_name: (model, y_pred, y_prob, metrics)}
        X_test, y_test: Test data
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison = []
    
    for model_name, (model, y_pred, y_prob, metrics) in models_dict.items():
        comparison.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'ROC-AUC': metrics['roc_auc'],
            'F1-Score': metrics['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
    
    return comparison_df


# ============================================================================
# MODEL EVALUATION & VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name='Model'):
    """
    Plot confusion matrix
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for title
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{model_name} Confusion Matrix Breakdown:")
    print(f"True Negatives:  {tn:,} (Correctly predicted not churned)")
    print(f"False Positives: {fp:,} (Incorrectly predicted churned)")
    print(f"False Negatives: {fn:,} (Missed churn cases)")
    print(f"True Positives:  {tp:,} (Correctly predicted churned)")
    print(f"\nPrecision: {tp/(tp+fp):.3f}")
    print(f"Recall: {tp/(tp+fn):.3f}")


def plot_roc_curve(y_test, y_prob, model_name='Model'):
    """
    Plot ROC curve
    
    Args:
        y_test: True labels
        y_prob: Predicted probabilities
        model_name: Name for legend
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance for tree-based models
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature importance")
        return
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top N
    top_features = importances.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'], color='#2E86AB', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'].tolist())
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importances


def plot_precision_recall_curve(y_test, y_prob, model_name='Model'):
    """
    Plot Precision-Recall curve
    
    Args:
        y_test: True labels
        y_prob: Predicted probabilities
        model_name: Name for title
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, color='#A23B72')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_classification_report(y_test, y_pred, model_name='Model'):
    """
    Print detailed classification report
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name for header
    """
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print('='*60)
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Churned', 'Churned'],
                                digits=3))


# ============================================================================
# CHURN RISK SCORING
# ============================================================================

def calculate_churn_risk_scores(model, features_df, X_encoded, scaler):
    """
    Calculate churn probability for all customers
    
    Args:
        model: Trained model
        features_df: Original features dataframe with customer_id
        X_encoded: Encoded features (without customer_id and target)
        scaler: Fitted scaler
        
    Returns:
        pd.DataFrame: Customer churn risk scores
    """
    # Scale features
    X_scaled = scaler.transform(X_encoded)
    
    # Predict probabilities
    churn_probs = model.predict_proba(X_scaled)[:, 1]
    
    # Create results dataframe
    risk_scores = pd.DataFrame({
        'customer_id': features_df['customer_id'],
        'churn_probability': churn_probs,
        'actual_churn': features_df['is_churned']
    })
    
    # Assign risk categories
    risk_scores['risk_category'] = pd.cut(
        risk_scores['churn_probability'],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
    )
    
    risk_scores = risk_scores.sort_values('churn_probability', ascending=False)
    
    return risk_scores


def segment_by_churn_risk(risk_scores, features_df):
    """
    Combine churn risk with customer features for actionable segmentation
    
    Args:
        risk_scores (pd.DataFrame): Churn risk scores
        features_df (pd.DataFrame): Customer features
        
    Returns:
        pd.DataFrame: Combined customer profiles with risk
    """
    # Merge risk scores with features
    customer_profiles = risk_scores.merge(
        features_df[['customer_id', 'recency_days', 'frequency', 'monetary', 
                    'avg_order_value', 'customer_age_days']],
        on='customer_id'
    )
    
    return customer_profiles


# ============================================================================
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================

def generate_churn_insights(risk_scores, features_df):
    """
    Generate business insights from churn predictions
    
    Args:
        risk_scores (pd.DataFrame): Churn risk scores
        features_df (pd.DataFrame): Customer features
        
    Returns:
        dict: Key insights and metrics
    """
    customer_profiles = segment_by_churn_risk(risk_scores, features_df)
    
    insights = {
        'total_customers': len(risk_scores),
        'predicted_churners': (risk_scores['churn_probability'] > 0.5).sum(),
        'churn_rate': (risk_scores['churn_probability'] > 0.5).sum() / len(risk_scores),
        'high_risk_count': (risk_scores['risk_category'].isin(['High Risk', 'Critical Risk'])).sum(),
        'revenue_at_risk': customer_profiles[
            customer_profiles['risk_category'].isin(['High Risk', 'Critical Risk'])
        ]['monetary'].sum()
    }
    
    # Risk distribution
    insights['risk_distribution'] = risk_scores['risk_category'].value_counts().to_dict()
    
    return insights