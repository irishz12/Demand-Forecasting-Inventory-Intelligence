FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src ./src
COPY data/processed/model_data.parquet ./data/processed/model_data.parquet
COPY models/demand_forecaster_xgboost.json ./models/demand_forecaster_xgboost.json
COPY data/raw/calendar.csv ./data/raw/calendar.csv

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
