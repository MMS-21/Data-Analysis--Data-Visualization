"""
Database utilities for SQLite operations
"""
import sqlite3
import pandas as pd
from pathlib import Path
import os

def create_database_schema(db_path):
    """
    Create SQLite database schema with normalized tables
    
    Args:
        db_path (str): Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create transactions table (main fact table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_no TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            description TEXT,
            quantity INTEGER NOT NULL,
            invoice_date TIMESTAMP NOT NULL,
            unit_price REAL NOT NULL,
            customer_id INTEGER NOT NULL,
            country TEXT NOT NULL,
            total_price REAL NOT NULL,
            year INTEGER,
            month INTEGER,
            year_month TEXT,
            day_of_week TEXT
        )
    """)
    
    # Create customers dimension table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            country TEXT NOT NULL,
            first_purchase_date TIMESTAMP,
            last_purchase_date TIMESTAMP,
            total_transactions INTEGER,
            total_revenue REAL,
            avg_order_value REAL
        )
    """)
    
    # Create products dimension table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            stock_code TEXT PRIMARY KEY,
            description TEXT,
            avg_unit_price REAL,
            total_quantity_sold INTEGER,
            total_revenue REAL,
            num_transactions INTEGER
        )
    """)
    
    # Create indexes for performance
    print("Creating indexes...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_customer 
        ON transactions(customer_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_date 
        ON transactions(invoice_date)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_stock 
        ON transactions(stock_code)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_invoice 
        ON transactions(invoice_no)
    """)
    
    conn.commit()
    
    # Verify indexes were created
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
    indexes = cursor.fetchall()
    print(f"Created {len(indexes)} indexes: {[idx[0] for idx in indexes]}")
    
    conn.close()
    
    print(f"✅ Database schema created at: {db_path}")


def load_data_to_db(df, db_path):
    """
    Load cleaned dataframe into SQLite database
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        db_path (str): Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    
    # Prepare transactions data
    transactions_df = df[[
        'InvoiceNo', 'StockCode', 'Description', 'Quantity', 
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country',
        'TotalPrice', 'Year', 'Month', 'YearMonth', 'DayOfWeek'
    ]].copy()
    
    transactions_df.columns = [
        'invoice_no', 'stock_code', 'description', 'quantity',
        'invoice_date', 'unit_price', 'customer_id', 'country',
        'total_price', 'year', 'month', 'year_month', 'day_of_week'
    ]
    
    # Load transactions
    print("Loading transactions table...")
    transactions_df.to_sql('transactions', conn, if_exists='replace', index=False)
    
    # Create customers dimension
    print("Creating customers dimension...")
    customers_df = df.groupby('CustomerID').agg({
        'Country': 'first',
        'InvoiceDate': ['min', 'max'],
        'InvoiceNo': 'nunique',
        'TotalPrice': ['sum', 'mean']
    }).reset_index()
    
    customers_df.columns = [
        'customer_id', 'country', 'first_purchase_date', 
        'last_purchase_date', 'total_transactions', 
        'total_revenue', 'avg_order_value'
    ]
    
    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    
    # Create products dimension
    print("Creating products dimension...")
    products_df = df.groupby('StockCode').agg({
        'Description': 'first',
        'UnitPrice': 'mean',
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique'
    }).reset_index()
    
    products_df.columns = [
        'stock_code', 'description', 'avg_unit_price',
        'total_quantity_sold', 'total_revenue', 'num_transactions'
    ]
    
    products_df.to_sql('products', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"✅ Data loaded successfully!")
    print(f"   - Transactions: {len(transactions_df):,} rows")
    print(f"   - Customers: {len(customers_df):,} rows")
    print(f"   - Products: {len(products_df):,} rows")


def verify_database(db_path):
    """
    Verify database integrity and print summary
    
    Args:
        db_path (str): Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    
    print("\n=== Database Verification ===")
    
    # Check row counts
    tables = ['transactions', 'customers', 'products']
    for table in tables:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        print(f"{table.capitalize()}: {count:,} rows")
    
    # Sample queries
    print("\n--- Sample Queries ---")
    
    # Top 5 customers by revenue
    top_customers = pd.read_sql("""
        SELECT customer_id, country, total_revenue, total_transactions
        FROM customers
        ORDER BY total_revenue DESC
        LIMIT 5
    """, conn)
    print("\nTop 5 Customers by Revenue:")
    print(top_customers.to_string(index=False))
    
    # Top 5 products by revenue
    top_products = pd.read_sql("""
        SELECT stock_code, description, total_revenue, total_quantity_sold
        FROM products
        ORDER BY total_revenue DESC
        LIMIT 5
    """, conn)
    print("\nTop 5 Products by Revenue:")
    print(top_products.to_string(index=False))
    
    # Monthly revenue trend
    monthly_revenue = pd.read_sql("""
        SELECT year_month, 
               SUM(total_price) as revenue,
               COUNT(DISTINCT customer_id) as unique_customers
        FROM transactions
        GROUP BY year_month
        ORDER BY year_month
    """, conn)
    print(f"\nMonthly data points: {len(monthly_revenue)}")
    
    conn.close()
    print("\n✅ Database verification complete!")


def execute_query(db_path, query):
    """
    Execute a SQL query and return results as DataFrame
    
    Args:
        db_path (str): Path to SQLite database
        query (str): SQL query string
        
    Returns:
        pd.DataFrame: Query results
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def main():
    """
    Main database setup pipeline
    """
    cwd = os.getcwd()
    DB_PATH = os.path.join(cwd, 'db', 'retail.db')
    CLEANED_DATA_PATH = os.path.join(cwd, 'data', 'cleaned', 'retail_clean.csv')
    
    # Create db directory if it doesn't exist
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['InvoiceDate'])
    print(f"Loaded {len(df):,} rows")
    
    # Create schema
    print("\nCreating database schema...")
    create_database_schema(DB_PATH)
    
    # Load data
    print("\nLoading data into database...")
    load_data_to_db(df, DB_PATH)
    
    # Verify
    print("\nVerifying database...")
    verify_database(DB_PATH)
    
    print("\n✅ Database setup complete!")


if __name__ == "__main__":
    main()