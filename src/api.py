from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.forecast import (
    load_data,
    load_model,
    load_calendar_lookup,
)
from src.inventory.inventory_recommendation import (
    generate_inventory_recommendation,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model, processed dataset, and calendar lookup once at startup
    load_data()
    load_model()
    load_calendar_lookup()
    yield


app = FastAPI(
    title="Demand Forecasting & Inventory Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    store_id: str
    item_id: str
    available_inventory: float = Field(ge=0)
    forecast_days: Literal[7, 14, 30] = 7


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "demand-forecasting-api",
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):

    try:
        result, forecast_df = generate_inventory_recommendation(
            store_id=request.store_id,
            item_id=request.item_id,
            available_inventory=request.available_inventory,
            forecast_days=request.forecast_days,
        )

        forecast_records = forecast_df.copy()

        forecast_records["date"] = (
            forecast_records["date"]
            .dt.strftime("%Y-%m-%d")
        )

        forecast_records = forecast_records[
            ["date", "forecast"]
        ].to_dict(orient="records")

        return {
            "store_id": request.store_id,
            "item_id": request.item_id,
            "forecast_days": request.forecast_days,
            "daily_forecast": forecast_records,
            "inventory": {
                "forecast_demand": result["forecast_demand"],
                "available_inventory": result["available_inventory"],
                "safety_stock": result["safety_stock"],
                "recommended_order": result["recommended_order"],
            },
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
