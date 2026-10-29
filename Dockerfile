FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY sql/ ./sql/
RUN pip install . && useradd --create-home appuser
COPY .streamlit/ ./.streamlit/
RUN mkdir -p data/features models mlruns reports && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
CMD ["uvicorn", "forecastops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
