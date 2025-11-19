import streamlit as st
import pandas as pd
st.write("mani")
# Direct download link from Google Drive

# Cache the data to avoid reloading on every app interaction
@st.cache_data
def load_data():
    return pd.read_parquet("df_cleaned_modeling.parquet", engine='pyarrow')  # or engine='fastparquet'

df = load_data()

st.write(df.head())