import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Direct download link from Google Drive

# Cache the data to avoid reloading on every app interaction
@st.cache_data
def load_data():
    return pd.read_parquet("dataset/df_cleaned_modeling.parquet", engine='pyarrow')  # or engine='fastparquet'

@st.cache_data
def load_lgbm():
    return joblib.load('models/LGBMRegressor (1).pkl')

@st.cache_data
def load_dec_tree():
    return joblib.load('models/DecisionTreeRegressor (1).pkl')

@st.cache_data
def load_lin_reg():
    return joblib.load('models/LinearRegression (1).pkl')

@st.cache_data
def load_xgb():
    return joblib.load('models/XGBRegressor (1).pkl')

@st.cache_data
def load_preprocessor():
    return joblib.load("preprocessor/scaler.pkl")

df = load_data()

lgbm = load_lgbm()
dec_tree = load_dec_tree()
lin_reg = load_lin_reg()
xgb = load_xgb()

preprocessor = load_preprocessor()

st.title("Sales Prediction Dashboard")
st.header("Select month and store")


col1, col2 = st.columns(2)
store = df['store_nbr'].unique()
month = df['month'].unique()

with col1:
    store_selected = st.selectbox(
    "Select Store Number",
    (sorted(store))
    )   
with col2:
    month_selected = st.selectbox(
    "Select Month Number",
    (sorted(month))
    )   

isHoliday_selected = st.radio(
    "Choose if there was holiday",
    ['Holiday', 'Not Holiday'],
    horizontal=True
)

isHoliday = 1 if isHoliday_selected=="Holiday" else 0

original = df[
    (df['store_nbr'] == store_selected) &
    (df['month'] == month_selected) & 
    (df['isHoliday'] == isHoliday)
]
features = ['store_nbr','isHoliday','month']
input_data = original[features]

input_scaled = preprocessor.transform(input_data)

pred_lgbm = lgbm.predict(input_scaled)
pred_dec_tree = dec_tree.predict(input_scaled)
pred_lin_reg = lin_reg.predict(input_scaled)
pred_xgb = xgb.predict(input_scaled)


# Add predictions to original dataframe
original['predicted_sales_lgbm'] = pred_lgbm
original['predicted_sales_dec_tree'] = pred_dec_tree
original['predicted_sales_lin_reg'] = pred_lin_reg
original['predicted_sales_xgb'] = pred_xgb


# Select first 10 rows
df_plot = original.head(10)

# Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_plot.index, df_plot['sales'], label='Actual', marker='o')
ax.plot(df_plot.index, df_plot['predicted_sales_lgbm'], label='LGBMRegressor', marker='x')
ax.plot(df_plot.index, df_plot['predicted_sales_dec_tree'], label='DecisionTreeRegressor', marker='x')
ax.plot(df_plot.index, df_plot['predicted_sales_lin_reg'], label='LinearRegression', marker='x')
ax.plot(df_plot.index, df_plot['predicted_sales_xgb'], label='XGBRegressor', marker='x')
ax.set_xlabel('Row Index')
ax.set_ylabel('Sales')
ax.set_title(f'Actual vs Predicted Sales (First 10 rows) for Store {store_selected}, Month {month_selected}')
ax.legend()

st.pyplot(fig)

st.header(
    "Flat prediction lines show the models aren't capturing sales variability.\n"
    "Possible reasons:\n"
    "- Missing important features\n"
    "- Model trained on too few features\n"
    "- Overfitting\n"
    "- Sales patterns too complex for current models"
)


