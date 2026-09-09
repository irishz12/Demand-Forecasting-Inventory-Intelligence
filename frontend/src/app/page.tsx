"use client";

import { useState } from "react";
import {
  BarChart3,
  Boxes,
  PackageCheck,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

type ForecastPoint = {
  date: string;
  forecast: number;
};

type ForecastResponse = {
  store_id: string;
  item_id: string;
  forecast_days: number;
  daily_forecast: ForecastPoint[];
  inventory: {
    forecast_demand: number;
    available_inventory: number;
    safety_stock: number;
    recommended_order: number;
  };
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") ||
  "http://localhost:8000";

export default function Home() {
  const [storeId, setStoreId] = useState("CA_1");
  const [itemId, setItemId] = useState("FOODS_3_090");
  const [inventory, setInventory] = useState("50");
  const [forecastDays, setForecastDays] = useState("7");

  const [result, setResult] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function generateForecast() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/forecast`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          store_id: storeId,
          item_id: itemId,
          available_inventory: Number(inventory),
          forecast_days: Number(forecastDays),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Forecast request failed");
      }

      const data: ForecastResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to forecasting API."
      );
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    result?.daily_forecast.map((point) => ({
      date: new Date(`${point.date}T00:00:00`).toLocaleDateString("en-US", {
        weekday: "short",
        month: "2-digit",
        day: "2-digit",
      }),
      forecast: Number(point.forecast.toFixed(2)),
    })) ?? [];

  const demandTrend = (() => {
    if (!result || result.daily_forecast.length < 2) return "—";

    const first = result.daily_forecast[0].forecast;
    const last =
      result.daily_forecast[result.daily_forecast.length - 1].forecast;

    const change = (last - first) / Math.max(first, 1);

    if (change > 0.05) return "Increasing";
    if (change < -0.05) return "Decreasing";
    return "Stable";
  })();

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl px-6 py-8">

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <div className="rounded-xl border p-2">
              <TrendingUp className="h-6 w-6" />
            </div>

            <div>
              <h1 className="text-2xl font-semibold tracking-tight">
                Demand Forecasting & Inventory Intelligence
              </h1>

              <p className="text-sm text-muted-foreground">
                AI-powered demand forecasting and inventory recommendations
              </p>
            </div>
          </div>

          <div className="mt-2 flex items-center gap-2">
            <Badge variant="secondary">XGBoost</Badge>
            <Badge variant="secondary">MLflow</Badge>
            <Badge variant="secondary">FastAPI</Badge>
          </div>
        </div>

        <Separator className="my-8" />

        <section className="grid gap-6 lg:grid-cols-[320px_1fr]">

          <Card>
            <CardHeader>
              <CardTitle>Forecast Controls</CardTitle>
            </CardHeader>

            <CardContent className="space-y-5">

              <div className="space-y-2">
                <label className="text-sm font-medium">Store</label>

                <Select
                  value={storeId}
                  onValueChange={(value) => value && setStoreId(value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>

                  <SelectContent>
                    <SelectItem value="CA_1">CA_1</SelectItem>
                    <SelectItem value="CA_2">CA_2</SelectItem>
                    <SelectItem value="TX_1">TX_1</SelectItem>
                    <SelectItem value="WI_1">WI_1</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Product</label>

                <Input
                  value={itemId}
                  onChange={(e) => setItemId(e.target.value)}
                  placeholder="FOODS_3_090"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Available Inventory
                </label>

                <Input
                  type="number"
                  min="0"
                  value={inventory}
                  onChange={(e) => setInventory(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Forecast Horizon
                </label>

                <Select
                  value={forecastDays}
                  onValueChange={(value) => value && setForecastDays(value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>

                  <SelectContent>
                    <SelectItem value="7">7 Days</SelectItem>
                    <SelectItem value="14">14 Days</SelectItem>
                    <SelectItem value="30">30 Days</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                className="w-full"
                onClick={generateForecast}
                disabled={loading}
              >
                {loading ? "Generating..." : "Generate Forecast"}
              </Button>

              {error && (
                <div className="rounded-lg border p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

            </CardContent>
          </Card>

          <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Forecast Demand
                </CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>

              <CardContent>
                <div className="text-2xl font-semibold">
                  {result
                    ? result.inventory.forecast_demand.toFixed(2)
                    : "—"}
                </div>

                <p className="text-xs text-muted-foreground">
                  Units over forecast horizon
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Available Inventory
                </CardTitle>
                <Boxes className="h-4 w-4 text-muted-foreground" />
              </CardHeader>

              <CardContent>
                <div className="text-2xl font-semibold">
                  {result
                    ? result.inventory.available_inventory
                    : inventory}
                </div>

                <p className="text-xs text-muted-foreground">
                  Current inventory
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Safety Stock
                </CardTitle>
                <PackageCheck className="h-4 w-4 text-muted-foreground" />
              </CardHeader>

              <CardContent>
                <div className="text-2xl font-semibold">
                  {result
                    ? result.inventory.safety_stock.toFixed(2)
                    : "—"}
                </div>

                <p className="text-xs text-muted-foreground">
                  Recommended buffer
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Recommended Order
                </CardTitle>

                {result && (
                  <Badge
                    variant={
                      result.inventory.recommended_order > 0
                        ? "destructive"
                        : "secondary"
                    }
                  >
                    {result.inventory.recommended_order > 0
                      ? "Reorder Required"
                      : "Stock Sufficient"}
                  </Badge>
                )}
              </CardHeader>

              <CardContent>
                <div className="text-2xl font-semibold">
                  {result
                    ? result.inventory.recommended_order.toFixed(2)
                    : "—"}
                </div>

                <p className="text-xs text-muted-foreground">
                  Units to replenish
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">
                  Demand Trend
                </CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>

              <CardContent>
                <div className="text-2xl font-semibold">
                  {demandTrend}
                </div>

                <p className="text-xs text-muted-foreground">
                  Across forecast horizon
                </p>
              </CardContent>
            </Card>

          </div>
        </section>

        <section className="mt-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Demand Forecast</CardTitle>
                <p className="mt-1 text-sm text-muted-foreground">
                  {result
                    ? `${result.forecast_days}-day forecast`
                    : "Generate a forecast to view predictions"}
                </p>
              </div>

              {result && (
                <div className="text-right">
                  <p className="text-xs text-muted-foreground">
                    Total Expected Demand
                  </p>
                  <p className="text-lg font-semibold">
                    {result.inventory.forecast_demand.toFixed(2)} units
                  </p>
                </div>
              )}
            </CardHeader>

            <CardContent>

              {!result ? (
                <div className="flex h-80 items-center justify-center rounded-lg border border-dashed">
                  <div className="text-center">
                    <BarChart3 className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />

                    <p className="font-medium">
                      Forecast visualization
                    </p>

                    <p className="text-sm text-muted-foreground">
                      Generate a forecast to view predictions.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />

                      <XAxis dataKey="date" />

                      <YAxis />

                      <Tooltip />

                      <Bar
                        dataKey="forecast"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

            </CardContent>
          </Card>
        </section>

      </div>
    </main>
  );
}
