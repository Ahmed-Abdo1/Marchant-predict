import pickle
import streamlit as st
import numpy as np
from PIL import Image

# Load model
model = pickle.load(open("F:/machine project/final project/XGBoost_best_model.pkl", "rb"))

st.set_page_config(page_title="Merchant Feature Prediction", page_icon="🧠", layout="wide")

with st.sidebar:
    st.image(r"F:\machine project\final project\proAr.png", width=150)
    st.title("🧠 Predictor App")
    page = st.radio("Navigate", ["🏠 Home", "📊 Predict"])
    st.markdown("---")
    st.write("Streamlit App for Predicting Merchant Loyalty Score")

feature_names = [
    'trans_purchase_year_mode', 'trans_purchase_month_mean', 'trans_purchase_month_mode',
    'trans_purchase_day_mode', 'trans_purchase_dow_mode', 'trans_purchase_hour_min',
    'trans_purchase_hour_max', 'trans_purchase_hour_mode', 'trans_is_weekend_mean',
    'trans_is_weekend_sum', 'feature_min', 'feature_std', 'feature_range', 'feature_sum_x_days',
    'days_feature_1', 'days_feature_1_ratio', 'days_feature_2', 'days_feature_2_ratio',
    'days_feature_3', 'days_feature_3_ratio'
]

if page == "🏠 Home":
    st.title("🧠 Merchant Loyalty Score Prediction App")
    st.markdown("""
    Welcome to the **Merchant Loyalty Prediction App**.  
    The goal of this project is to **predict customer loyalty scores**
    for merchants based on their **transactional behavior**.

    This is a **regression task** where we estimate how loyal a customer is,
    using features like transaction frequency, timing, and activity stats.

    ### 📌 What you can do:
    - Enter merchant numerical and categorical features
    - Use sliders and inputs for easy control
    - Predict with one click!
    """)
    st.image(r"F:\machine project\final project\proAr.png", width=250)

if page == "📊 Predict":
    st.title("📊 Feature Inputs")

    st.subheader("🔢 Enter Numeric Features")
    inputs = []
    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        val = col.number_input(f"{feature}", value=0.0, step=0.01, format="%.4f")
        inputs.append(val)

    st.subheader("🗂️ Categorical Features")
    category_col1, category_col2 = st.columns(2)
    category_1 = category_col1.radio("category_1 (Y/N)", ["Y", "N"])
    category_4 = category_col2.radio("category_4 (Y/N)", ["Y", "N"])

    category_1_encoded = 1 if category_1 == 'Y' else 0
    category_4_encoded = 1 if category_4 == 'Y' else 0

    col3, col4 = st.columns(2)
    most_recent_sales_range = col3.selectbox("most_recent_sales_range", ["A", "B", "C", "D", "E"])
    most_recent_purchases_range = col4.selectbox("most_recent_purchases_range", ["A", "B", "C", "D", "E"])

    sales_range_encoded = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}[most_recent_sales_range]
    purchase_range_encoded = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}[most_recent_purchases_range]

    # Combine all features
    full_features = inputs + [category_1_encoded, category_4_encoded, sales_range_encoded, purchase_range_encoded]

    st.markdown("----")
    if st.button("🔮 Predict Loyalty Score"):
        try:
            input_array = np.array(full_features).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            st.success(f"✅ Predicted Loyalty Score: {prediction:.4f}")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
