import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

print("\n📊 Sales Forecasting Project Started\n")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ---------------- MONTHLY SALES ----------------
monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()
plt.show()

# ---------------- CATEGORY SALES ----------------
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(6,4))
category_sales.plot(kind="bar")
plt.title("Category Wise Sales")
plt.ylabel("Sales")
plt.grid()
plt.show()

# ---------------- FORECAST MODEL ----------------
sales_data = df.groupby("Order Date")["Sales"].sum().reset_index()
sales_data.columns = ["ds", "y"]

model = Prophet()
model.fit(sales_data)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# ---------------- FORECAST GRAPH ----------------
model.plot(forecast)
plt.title("Sales Forecast (Next 365 Days)")
plt.show()

# ---------------- SAVE OUTPUT ----------------
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    "sales_forecast.csv",
    index=False
)

print("\n✅ Forecast Completed Successfully")
print("📁 sales_forecast.csv generated")