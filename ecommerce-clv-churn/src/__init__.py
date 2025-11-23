"""
E-commerce CLV & Churn Analytics Package
Initialization file for easy imports
"""

# Import from data_utils
from .data_utils import (
    load_raw_data,
    clean_retail_data,
    generate_summary_stats,
    save_cleaned_data
)

# Import from db_utils
from .db_utils import (
    create_database_schema,
    load_data_to_db,
    verify_database,
    execute_query
)

# Import from analysis_utils
from .analysis_utils import (
    # Product analysis
    get_product_details,
    get_top_products,
    analyze_product_trend,
    
    # Customer analysis
    get_customer_profile,
    get_customer_summary,
    segment_customers_simple,
    
    # Time series analysis
    get_monthly_metrics,
    calculate_growth_rates,
    create_cohort_analysis,
    
    # Visualization helpers
    plot_top_items,
    plot_time_series,
    
    # Reporting
    export_summary_report,
    print_business_summary
)

# Version info
__version__ = '1.0.0'
__author__ = 'Your Name'

# Make commonly used functions easily accessible
__all__ = [
    # Data utilities
    'load_raw_data',
    'clean_retail_data',
    'generate_summary_stats',
    'save_cleaned_data',
    
    # Database utilities
    'create_database_schema',
    'load_data_to_db',
    'verify_database',
    'execute_query',
    
    # Product analysis
    'get_product_details',
    'get_top_products',
    'analyze_product_trend',
    
    # Customer analysis
    'get_customer_profile',
    'get_customer_summary',
    'segment_customers_simple',
    
    # Time series
    'get_monthly_metrics',
    'calculate_growth_rates',
    'create_cohort_analysis',
    
    # Visualization
    'plot_top_items',
    'plot_time_series',
    
    # Reporting
    'export_summary_report',
    'print_business_summary'
]

print(f"✅ E-commerce Analytics Package v{__version__} loaded successfully")