# ForecastOps Makefile

.PHONY: help install data train tune evaluate benchmark api dashboard test lint format

# Default target
help:
	@echo "ForecastOps - Retail Demand Forecasting Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     Install Python dependencies"
	@echo "  make data        Download and prepare M5 dataset"
	@echo "  make train       Train a model"
	@echo "  make tune        Tune model hyperparameters"
	@echo "  make evaluate    Evaluate models"
	@echo "  make benchmark   Benchmark model performance"
	@echo "  make api         Start FastAPI server"
	@echo "  make dashboard   Start Streamlit dashboard"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linter"
	@echo "  make format      Format code"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -e .

# Download and prepare data
data:
	python scripts/download_data.py --output data/raw
	python scripts/prepare_data.py --data-dir data/raw --subset-products 100 --subset-stores 5 --subset-days 730

# Train models
train:
	python scripts/build_retail_plan.py

tune: tune-xgboost

train-xgboost:
	python scripts/train.py --model xgboost

train-lightgbm:
	python scripts/train.py --model lightgbm

train-catboost:
	python scripts/train.py --model catboost

train-naive:
	python scripts/train.py --model naive

train-linear:
	python scripts/train.py --model linear

# Tune models
tune-xgboost:
	python scripts/tune.py --model xgboost --n-trials 50

tune-lightgbm:
	python scripts/tune.py --model lightgbm --n-trials 50

# Evaluate models
evaluate:
	python scripts/evaluate.py

# Benchmark models
benchmark:
	python scripts/benchmark.py

# Start API server
api:
	uvicorn forecastops.api.main:app --reload --host 0.0.0.0 --port 8000

# Start dashboard
dashboard:
	streamlit run src/forecastops/dashboard/streamlit_app.py

# Run tests
test:
	pytest tests/ -v --tb=short

# Run linter
lint:
	ruff check src/forecastops/ tests/

# Format code
format:
	ruff format src/forecastops/ tests/

# Check types
check-types:
	mypy src/forecastops/

# Clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf data/features
	rm -rf models
	rm -rf mlruns
	rm -rf reports
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Docker commands
docker-build:
	docker build -t forecastops:latest .

docker-run:
	docker run -it --rm -p 8000:8000 -p 8501:8501 forecastops:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down
