import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

print("\n📊 Project Started Successfully\n")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ---------------- BASIC INFO ----------------
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# ---------------- DAILY SALES ----------------
daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index()

plt.figure(figsize=(10,5))
plt.plot(daily_sales["Order Date"], daily_sales["Sales"])
plt.title("📈 Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()

# ---------------- MONTHLY SALES ----------------
df["Month"] = df["Order Date"].dt.to_period("M")
monthly_sales = df.groupby("Month")["Sales"].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("📊 Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid()
plt.show()

# ---------------- CATEGORY ANALYSIS ----------------
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(6,4))
category_sales.plot(kind="bar")
plt.title("📦 Category Wise Sales")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.grid()
plt.show()

# ---------------- FORECAST MODEL ----------------
print("\n🔮 Training Forecast Model...\n")

sales_data = df.groupby("Order Date")["Sales"].sum().reset_index()
sales_data.columns = ["ds", "y"]

model = Prophet()
model.fit(sales_data)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# ---------------- FORECAST RESULT ----------------
print("\n📊 Forecast Sample:\n")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# ---------------- FINAL FORECAST GRAPH ----------------
fig = model.plot(forecast)
plt.title("🔮 Sales Forecast (Next 365 Days)")
plt.show()

# ---------------- SAVE OUTPUT ----------------
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    "sales_forecast.csv",
    index=False
)

print("\n✅ Project Completed Successfully!")
print("📁 File saved: sales_forecast.csv")