import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ---------------- PAGE SETTINGS ---------------- #
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("### Data Science Internship Project")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df

df = load_data()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category))
]

# ---------------- KPI CARDS ---------------- #
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
avg_sales = filtered_df["Sales"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Sales", f"${total_sales:,.2f}")
c2.metric("📈 Total Profit", f"${total_profit:,.2f}")
c3.metric("🛒 Orders", total_orders)
c4.metric("📦 Avg Sales", f"${avg_sales:,.2f}")

st.divider()

# ---------------- SALES TREND ---------------- #
daily_sales = (
    filtered_df
    .groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
)

fig = px.line(
    daily_sales,
    x="Order Date",
    y="Sales",
    title="📈 Daily Sales Trend"
)

st.plotly_chart(fig, use_container_width=True)