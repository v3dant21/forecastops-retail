PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS retail_imports (
    source_sha256 TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    source_sheet TEXT NOT NULL,
    repeated_headers INTEGER NOT NULL,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS retail_customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS retail_products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    unit TEXT NOT NULL,
    hsn_code TEXT NOT NULL,
    scope TEXT NOT NULL CHECK(scope IN ('food', 'non_food', 'unclassified'))
);
CREATE TABLE IF NOT EXISTS retail_invoices (
    invoice_key TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL,
    date TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES retail_customers(customer_id),
    source_sha256 TEXT NOT NULL REFERENCES retail_imports(source_sha256),
    source_invoice_row INTEGER NOT NULL,
    invoice_quantity REAL,
    invoice_taxable_paise INTEGER,
    invoice_tax_paise INTEGER,
    invoice_total_paise INTEGER,
    invoice_grand_total_paise INTEGER,
    content_sha256 TEXT NOT NULL,
    UNIQUE(date, invoice_id)
);
CREATE TABLE IF NOT EXISTS retail_invoice_sources (
    source_sha256 TEXT NOT NULL REFERENCES retail_imports(source_sha256),
    invoice_key TEXT NOT NULL REFERENCES retail_invoices(invoice_key),
    source_invoice_row INTEGER NOT NULL,
    PRIMARY KEY(source_sha256, invoice_key)
);
CREATE TABLE IF NOT EXISTS retail_invoice_lines (
    invoice_key TEXT NOT NULL REFERENCES retail_invoices(invoice_key),
    line_number INTEGER NOT NULL,
    product_id TEXT NOT NULL REFERENCES retail_products(product_id),
    source_row INTEGER NOT NULL,
    original_product_name TEXT NOT NULL,
    unit_original TEXT NOT NULL,
    quantity REAL,
    unit_rate REAL,
    taxable_paise INTEGER,
    tax_paise INTEGER,
    total_paise INTEGER,
    base_issues TEXT NOT NULL,
    PRIMARY KEY(invoice_key, line_number)
);
CREATE INDEX IF NOT EXISTS retail_invoice_date_idx ON retail_invoices(date);
CREATE INDEX IF NOT EXISTS retail_invoice_customer_idx ON retail_invoices(customer_id);
CREATE INDEX IF NOT EXISTS retail_line_product_idx ON retail_invoice_lines(product_id);

CREATE VIEW IF NOT EXISTS retail_food_sales AS
SELECT i.invoice_key, i.invoice_id, i.date, c.customer_name AS customer,
       i.source_sha256, l.line_number, l.source_row,
       p.product_id, p.product_name, p.unit, p.hsn_code,
       l.quantity, l.taxable_paise / 100.0 AS taxable_value,
       l.tax_paise / 100.0 AS tax_amount, l.total_paise / 100.0 AS item_total
FROM retail_invoice_lines l
JOIN retail_invoices i ON i.invoice_key = l.invoice_key
JOIN retail_products p ON p.product_id = l.product_id
JOIN retail_customers c ON c.customer_id = i.customer_id
WHERE p.scope = 'food';

CREATE VIEW IF NOT EXISTS retail_daily_food_sales AS
SELECT date, product_id, product_name, unit, SUM(quantity) AS quantity,
       SUM(taxable_value) AS taxable_value, SUM(tax_amount) AS tax_amount,
       SUM(item_total) AS item_total, COUNT(*) AS item_lines
FROM retail_food_sales
GROUP BY date, product_id, product_name, unit;
CREATE TABLE IF NOT EXISTS retail_resolutions (
    resolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK(kind IN ('coverage', 'units', 'product', 'line', 'invoice', 'duplicate')),
    target TEXT NOT NULL,
    values_json TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS retail_resolution_target_idx
ON retail_resolutions(kind, target, resolution_id);
PRAGMA user_version = 2;
