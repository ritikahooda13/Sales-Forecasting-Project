import streamlit as pd_st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime

# Page configuration
pd_st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

pd_st.title("📈 Sales Forecasting System")
pd_st.write("Analyze past sales data and predict future sales using Machine Learning.")

# 1. Data Generation
@pd_st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    base_sales = np.random.randint(200, 500, size=len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Sales': base_sales,
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home Appliances', 'Books'], size=len(dates))
    })
    return df

df = load_data().copy()

# Features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

col1, col2 = pd_st.columns(2)

with col1:
    pd_st.subheader("📊 Dataset Preview")
    pd_st.dataframe(df.head(15), use_container_width=True)

with col2:
    pd_st.subheader("🔮 Predict Future Sales")
    
    # Model Training
    X = df[['Year', 'Month', 'Day', 'DayOfWeek']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    input_date = pd_st.date_input("Select a Future Date to Forecast:", datetime.date(2026, 7, 1))
    
    if pd_st.button("Forecast Sales"):
        pred_features = pd.DataFrame({
            'Year': [input_date.year],
            'Month': [input_date.month],
            'Day': [input_date.day],
            'DayOfWeek': [input_date.weekday()]
        })
        predicted_value = model.predict(pred_features)[0]
        pd_st.success(f"🎯 Predicted Sales for {input_date}: **{predicted_value:.2f} units**")
