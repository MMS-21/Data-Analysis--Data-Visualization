"""
Tests for data ingestion and database integrity
"""
import pytest
import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = 'db/retail.db'
CLEANED_DATA_PATH = 'data/cleaned/retail_clean.csv'


@pytest.fixture
def db_connection():
    """Fixture to provide database connection"""
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture
def cleaned_df():
    """Fixture to provide cleaned dataframe"""
    df = pd.read_csv(CLEANED_DATA_PATH)
    return df


def test_database_exists():
    """Test that database file exists"""
    assert Path(DB_PATH).exists(), f"Database file not found at {DB_PATH}"


def test_tables_exist(db_connection):
    """Test that all required tables exist"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = ['transactions', 'customers', 'products']
    for table in required_tables:
        assert table in tables, f"Table '{table}' not found in database"


def test_transactions_row_count(db_connection, cleaned_df):
    """Test that transactions table has correct number of rows"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    db_count = cursor.fetchone()[0]
    
    # Should match cleaned CSV
    assert db_count == len(cleaned_df), f"Row count mismatch: DB has {db_count}, CSV has {len(cleaned_df)}"


def test_no_null_customer_ids(db_connection):
    """Test that there are no NULL customer IDs"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE customer_id IS NULL")
    null_count = cursor.fetchone()[0]
    
    assert null_count == 0, f"Found {null_count} NULL customer IDs"


def test_no_negative_values(db_connection):
    """Test that there are no negative quantities or prices"""
    cursor = db_connection.cursor()
    
    # Check negative quantities
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE quantity <= 0")
    neg_qty = cursor.fetchone()[0]
    assert neg_qty == 0, f"Found {neg_qty} negative/zero quantities"
    
    # Check negative prices
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE unit_price <= 0")
    neg_price = cursor.fetchone()[0]
    assert neg_price == 0, f"Found {neg_price} negative/zero prices"
    
    # Check negative total_price
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE total_price <= 0")
    neg_total = cursor.fetchone()[0]
    assert neg_total == 0, f"Found {neg_total} negative/zero total prices"


def test_data_types(db_connection):
    """Test that key columns have correct data types"""
    df = pd.read_sql("SELECT * FROM transactions LIMIT 5", db_connection)
    
    # Check that numeric columns are numeric
    assert pd.api.types.is_numeric_dtype(df['quantity']), "quantity should be numeric"
    assert pd.api.types.is_numeric_dtype(df['unit_price']), "unit_price should be numeric"
    assert pd.api.types.is_numeric_dtype(df['total_price']), "total_price should be numeric"
    assert pd.api.types.is_numeric_dtype(df['customer_id']), "customer_id should be numeric"


def test_unique_customers_count(db_connection, cleaned_df):
    """Test that customers table has correct number of unique customers"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    db_customer_count = cursor.fetchone()[0]
    
    csv_unique_customers = cleaned_df['CustomerID'].nunique()
    
    assert db_customer_count == csv_unique_customers, \
        f"Customer count mismatch: DB has {db_customer_count}, CSV has {csv_unique_customers}"


def test_revenue_aggregation(db_connection, cleaned_df):
    """Test that total revenue matches between transactions and aggregations"""
    cursor = db_connection.cursor()
    
    # Total from transactions
    cursor.execute("SELECT SUM(total_price) FROM transactions")
    db_total_revenue = cursor.fetchone()[0]
    
    # Total from CSV
    csv_total_revenue = cleaned_df['TotalPrice'].sum()
    
    # Allow small floating point differences
    assert abs(db_total_revenue - csv_total_revenue) < 0.01, \
        f"Revenue mismatch: DB has {db_total_revenue}, CSV has {csv_total_revenue}"


def test_customer_aggregations(db_connection):
    """Test that customer aggregations are reasonable"""
    df = pd.read_sql("SELECT * FROM customers", db_connection)
    
    # All customers should have at least 1 transaction
    assert (df['total_transactions'] >= 1).all(), "Some customers have 0 transactions"
    
    # All customers should have positive revenue
    assert (df['total_revenue'] > 0).all(), "Some customers have non-positive revenue"
    
    # First purchase should be before or equal to last purchase
    df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    assert (df['first_purchase_date'] <= df['last_purchase_date']).all(), \
        "Some customers have first_purchase_date after last_purchase_date"


def test_product_aggregations(db_connection):
    """Test that product aggregations are reasonable"""
    df = pd.read_sql("SELECT * FROM products", db_connection)
    
    # All products should have positive quantity sold
    assert (df['total_quantity_sold'] > 0).all(), "Some products have non-positive quantity"
    
    # All products should have positive revenue
    assert (df['total_revenue'] > 0).all(), "Some products have non-positive revenue"
    
    # All products should have at least 1 transaction
    assert (df['num_transactions'] >= 1).all(), "Some products have 0 transactions"


def test_indexes_exist(db_connection):
    """Test that required indexes exist"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
    indexes = [row[0] for row in cursor.fetchall()]
    
    # Check that at least some indexes were created
    assert len(indexes) >= 3, f"Expected at least 3 indexes, found {len(indexes)}: {indexes}"
    
    # Verify indexes contain expected patterns
    index_str = ' '.join(indexes)
    assert 'customer' in index_str.lower(), "No customer index found"
    assert 'date' in index_str.lower(), "No date index found"


def test_date_range(db_connection):
    """Test that date range is reasonable"""
    cursor = db_connection.cursor()
    cursor.execute("SELECT MIN(invoice_date), MAX(invoice_date) FROM transactions")
    min_date, max_date = cursor.fetchone()
    
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    
    # Dates should be in reasonable range (2009-2011 for Online Retail dataset)
    assert min_date.year >= 2009, f"Min date year {min_date.year} seems too early"
    assert max_date.year <= 2012, f"Max date year {max_date.year} seems too late"
    assert min_date < max_date, "Min date should be before max date"
    
    print(f"\n✓ Date range: {min_date.date()} to {max_date.date()}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, '-v'])