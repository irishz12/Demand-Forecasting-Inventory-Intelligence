# Demand Forecasting & Inventory Intelligence

An end-to-end retail demand forecasting and inventory recommendation system built with XGBoost, FastAPI, Next.js, Docker, Docker Compose, and MLflow.

## Problem

Retail inventory teams need accurate demand forecasts to reduce stockouts while avoiding unnecessary excess inventory.

This project forecasts product demand at the Store × Product level and converts the forecast into an inventory recommendation.

## Architecture

M5 Dataset
→ Data Preparation
→ Time-Series Feature Engineering
→ XGBoost Forecasting
→ SHAP Explainability
→ Recursive Forecasting
→ Inventory Recommendation
→ FastAPI
→ Next.js Dashboard

## Machine Learning

### Dataset

M5 Forecasting - Accuracy (Walmart)

The MVP currently evaluates the 50 highest-volume Store × Product series.

### Features

- Lag 1, 7, 14, 28 days
- Rolling mean and standard deviation
- Day of week
- Month
- Year
- Week of year
- Sell price
- SNAP indicators
- Event indicators

### Models

Baselines:

- Naive
- Seasonal Naive
- Moving Average

Production model:

- XGBoost Regressor

### Evaluation

Metrics:

- MAE
- RMSE
- WAPE

XGBoost validation result:

- MAE: 8.6744
- RMSE: 12.6023
- WAPE: 22.5481%

Best baseline WAPE:

- 29.2454%

Relative WAPE reduction:

- ~22.9%

## Explainability

SHAP is used to identify the most important forecasting features.

Top drivers in the validation analysis:

1. lag_1
2. rolling_mean_7
3. day_of_week

## Inventory Intelligence

The system converts forecasts into a replenishment recommendation:

`Recommended Order = Forecast Demand + Safety Stock - Available Inventory`

Safety stock is currently implemented as a configurable percentage of forecast demand.

The project also includes a decision-time inventory simulation comparing a simple baseline strategy against the forecast-driven strategy.

Simulation result on the selected 50 Store × Product series:

- Stockout rate: 9.0% → 3.5%
- Service level: 91.0% → 96.5%
- Inventory-related cost: 14.13 → 11.98%

These results come from an MVP simulation with explicit inventory-buffer and cost assumptions and should not be interpreted as a production replenishment optimizer.

## API

FastAPI provides:

- `GET /health`
- `POST /forecast`

Example request:

```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_3_090",
  "available_inventory": 50,
  "forecast_days": 7
}
Frontend
The dashboard is built with:
- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- Recharts
It provides:
- Store selection
- Product selection
- Inventory input
- Forecast horizon selection
- Forecast demand KPIs
- Safety stock
- Recommended order
- Demand trend
- Demand forecast chart
MLOps
The project includes:
- Git
- MLflow experiment tracking
- MLflow Model Registry
- Docker
- Docker Compose
- Reproducible production dependencies
- FastAPI serving
- Next.js frontend
Project Structure
.
├── data/
├── frontend/
├── models/
├── reports/
├── src/
│   ├── api.py
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── inference/
│   └── inventory/
├── Dockerfile
├── docker-compose.yml
├── requirements-dev.txt
├── requirements-prod.txt
└── README.md
Running the Application
Build the backend image:
docker build -t demand-forecasting-api:latest .
Build the frontend image:
cd frontend
docker build -t demand-forecasting-frontend:latest .
cd ..
Start the complete application:
docker-compose up
Frontend:
http://localhost:3000
API:
http://localhost:8000
API documentation:
http://localhost:8000/docs
Important Scope Notes
This is an MVP portfolio implementation.
Current limitations include:
- Forecasting is evaluated on the top 50 high-volume Store × Product series.
- Future price and event handling is simplified for recursive inference.
- Inventory optimization uses a simple safety-stock policy rather than a full stochastic optimization model.
- The business simulation uses explicit assumptions for inventory buffer and costs.
These limitations are documented intentionally to keep the system reproducible and avoid overstating production readiness.
