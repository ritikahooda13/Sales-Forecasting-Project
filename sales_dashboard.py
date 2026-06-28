import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("📊 Sales Forecast Dashboard")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# ---------------- FILTERS ----------------
st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Region", df["Region"].unique(), df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Category", df["Category"].unique(), df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]

# ---------------- KPI ----------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Sales", f"{filtered_df['Sales'].sum():,.2f}")
col2.metric("📈 Profit", f"{filtered_df['Profit'].sum():,.2f}")
col3.metric("🛒 Orders", filtered_df["Order ID"].nunique())

st.divider()

# ---------------- SALES TREND ----------------
daily_sales = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()

fig1 = px.line(daily_sales, x="Order Date", y="Sales", title="Sales Trend")
st.plotly_chart(fig1, use_container_width=True)

# ---------------- CATEGORY CHART ----------------
cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()

fig2 = px.bar(cat_sales, x="Category", y="Sales", title="Category Sales")
st.plotly_chart(fig2, use_container_width=True)

# ---------------- FORECAST ----------------
st.subheader("🔮 Forecast (Prophet)")

forecast_df = filtered_df.groupby("Order Date")["Sales"].sum().reset_index()
forecast_df.columns = ["ds", "y"]

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig3 = model.plot(forecast)
st.pyplot(fig3)

# ---------------- DOWNLOAD ----------------
csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)

st.download_button(
    "📥 Download Forecast CSV",
    csv,
    "sales_forecast.csv",
    "text/csv"
)

st.success("Dashboard Ready 🚀")