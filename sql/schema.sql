-- ForecastOps Database Schema
-- PostgreSQL database for M5 Forecasting data

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Stores table
CREATE TABLE IF NOT EXISTS stores (
    id SERIAL PRIMARY KEY,
    store_id VARCHAR(50) UNIQUE NOT NULL,
    store_name VARCHAR(255),
    state_id VARCHAR(50),
    state_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(100) UNIQUE NOT NULL,
    item_id VARCHAR(100),
    dept_id VARCHAR(50),
    dept_name VARCHAR(100),
    cat_id VARCHAR(50),
    cat_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Calendar table
CREATE TABLE IF NOT EXISTS calendar (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    d VARCHAR(10),
    wm_yr_wk INTEGER,
    weekday VARCHAR(10),
    wday INTEGER,
    month INTEGER,
    year INTEGER,
    snap_ca BOOLEAN DEFAULT FALSE,
    snap_tx BOOLEAN DEFAULT FALSE,
    snap_wi BOOLEAN DEFAULT FALSE,
    event_name_1 VARCHAR(100),
    event_type_1 VARCHAR(50),
    event_name_2 VARCHAR(100),
    event_type_2 VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prices table
CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(100) REFERENCES products(product_id),
    store_id VARCHAR(50) REFERENCES stores(store_id),
    wm_yr_wk INTEGER,
    sell_price NUMERIC(10, 4),
    price_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, store_id, wm_yr_wk)
);

-- Sales table
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(100) REFERENCES products(product_id),
    store_id VARCHAR(50) REFERENCES stores(store_id),
    d VARCHAR(10),
    date DATE,
    sales INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, store_id, d)
);

-- Features table (for pre-computed features)
CREATE TABLE IF NOT EXISTS sales_features (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(100),
    store_id VARCHAR(50),
    date DATE,
    d VARCHAR(10),
    -- Lag features
    sales_lag_1 INTEGER,
    sales_lag_2 INTEGER,
    sales_lag_3 INTEGER,
    sales_lag_7 INTEGER,
    sales_lag_14 INTEGER,
    sales_lag_21 INTEGER,
    sales_lag_28 INTEGER,
    -- Rolling features
    sales_rolling_mean_7 NUMERIC(12, 4),
    sales_rolling_mean_14 NUMERIC(12, 4),
    sales_rolling_mean_28 NUMERIC(12, 4),
    sales_rolling_std_7 NUMERIC(12, 4),
    sales_rolling_std_14 NUMERIC(12, 4),
    sales_rolling_std_28 NUMERIC(12, 4),
    -- Price features
    price_lag_1 NUMERIC(10, 4),
    price_change_1 NUMERIC(10, 4),
    price_change_7 NUMERIC(10, 4),
    -- Calendar features
    is_weekend BOOLEAN,
    is_month_start BOOLEAN,
    is_month_end BOOLEAN,
    is_holiday_season BOOLEAN,
    -- Target (can be updated with derived features)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(product_id, store_id, date)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sales_product_store ON sales(product_id, store_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_store ON sales(store_id);

CREATE INDEX IF NOT EXISTS idx_prices_product_store ON prices(product_id, store_id);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(price_date);

CREATE INDEX IF NOT EXISTS idx_features_product_store ON sales_features(product_id, store_id);
CREATE INDEX IF NOT EXISTS idx_features_date ON sales_features(date);

CREATE INDEX IF NOT EXISTS idx_calendar_date ON calendar(date);
CREATE INDEX IF NOT EXISTS idx_calendar_wm_yr_wk ON calendar(wm_yr_wk);

CREATE INDEX IF NOT EXISTS idx_products_dept ON products(dept_id);
CREATE INDEX IF NOT EXISTS idx_products_cat ON products(cat_id);

-- Views for common queries

-- Daily sales by store and category
CREATE OR REPLACE VIEW daily_sales_by_category AS
SELECT 
    s.date,
    p.cat_id,
    p.cat_name,
    s.store_id,
    SUM(s.sales) as total_sales,
    COUNT(DISTINCT s.product_id) as product_count
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY s.date, p.cat_id, p.cat_name, s.store_id
ORDER BY s.date, p.cat_id, s.store_id;

-- Sales trends (7-day rolling average)
CREATE OR REPLACE VIEW sales_trends AS
SELECT 
    s.date,
    s.product_id,
    s.store_id,
    s.sales,
    AVG(s.sales) OVER (
        PARTITION BY s.product_id, s.store_id 
        ORDER BY s.date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as sales_7day_avg
FROM sales s
ORDER BY s.product_id, s.store_id, s.date;

-- Price elasticity features
CREATE OR REPLACE VIEW price_features AS
SELECT 
    p.product_id,
    p.store_id,
    p.price_date,
    p.sell_price,
    LAG(p.sell_price, 1) OVER (
        PARTITION BY p.product_id, p.store_id 
        ORDER BY p.price_date
    ) as prev_price,
    (p.sell_price - LAG(p.sell_price, 1) OVER (
        PARTITION BY p.product_id, p.store_id 
        ORDER BY p.price_date
    )) / NULLIF(LAG(p.sell_price, 1) OVER (
        PARTITION BY p.product_id, p.store_id 
        ORDER BY p.price_date
    ), 0) as price_change_pct
FROM prices p
ORDER BY p.product_id, p.store_id, p.price_date;

-- Comments for documentation
COMMENT ON TABLE sales IS 'Daily sales data for products across stores';
COMMENT ON TABLE products IS 'Product catalog with category information';
COMMENT ON TABLE stores IS 'Store information including state';
COMMENT ON TABLE calendar IS 'Calendar data with dates and event information';
COMMENT ON TABLE prices IS 'Product prices over time';
COMMENT ON TABLE sales_features IS 'Pre-computed features for model training';

COMMENT ON COLUMN sales_features.sales_lag_1 IS 'Sales from 1 day ago';
COMMENT ON COLUMN sales_features.sales_lag_7 IS 'Sales from 7 days ago';
COMMENT ON COLUMN sales_features.sales_lag_28 IS 'Sales from 28 days ago';
COMMENT ON COLUMN sales_features.sales_rolling_mean_7 IS '7-day rolling average of sales';
COMMENT ON COLUMN sales_features.sales_rolling_std_7 IS '7-day rolling standard deviation of sales';
COMMENT ON COLUMN sales_features.price_lag_1 IS 'Price from 1 day ago';
COMMENT ON COLUMN sales_features.price_change_1 IS 'Price change from previous day';
COMMENT ON COLUMN sales_features.is_weekend IS 'Flag for weekend days (Saturday/Sunday)';
