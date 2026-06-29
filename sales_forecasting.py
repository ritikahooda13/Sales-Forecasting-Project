import streamlit as pd_st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

# Page configuration
pd_st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

pd_st.title("📈 Sales Forecasting System")
pd_st.write("Analyze past sales data and predict future sales using Machine Learning.")

# 1. Data Generation (Synthetic Dataset to avoid file path errors)
@pd_st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    base_sales = np.random.randint(200, 500, size=len(dates))
    
    # Adding some seasonality (higher sales on weekends)
    weekend_effect = [100 if d.weekday() >= 5 else 0 for d in dates]
    sales = base_sales + weekend_effect
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home Appliances', 'Books'], size=len(dates))
    })
    return df

df = load_data().copy()

# 2. Data Preprocessing & Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Layout columns for Dashboard
col1, col2 = pd_st.columns([1, 2])

with col1:
    pd_st.subheader("📊 Dataset Preview")
    pd_st.dataframe(df.head(10), use_container_width=True)
    
    # Model Training Setup
    X = df[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    pd_st.subheader("⚙️ Model Metrics")
    pd_st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    pd_st.metric(label="R² Score", value=f"{r2:.2f}")

with col2:
    pd_st.subheader("📉 Sales Trends (EDA)")
    
    # Monthly sales trend plot
    monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly_sales, x='Month', y='Sales', hue='Year', marker='o', palette='tab10', ax=ax)
    ax.set_title("Monthly Sales Trend Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    pd_st.pyplot(fig)
    
    # Future Prediction Form
    pd_st.subheader("🔮 Predict Future Sales")
    input_date = pd_st.date_input("Select a Future Date to Forecast Sales:", datetime.date(2026, 7, 1))
    
    if pd_st.button("Forecast Sales"):
        pred_features = pd.DataFrame({
            'Year': [input_date.year],
            'Month': [input_date.month],
            'Day': [input_date.day],
            'DayOfWeek': [input_date.weekday()]
        })
        predicted_value = model.predict(pred_features)[0]
        pd_st.success(f"🎯 Predicted Sales for {input_date}: **{predicted_value:.2f} units**")