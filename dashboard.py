import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("🚀 Corporate R&D Innovation Dashboard")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # (your preprocessing goes here)
    return df

df = load_data("panel_2015_2018.csv")

st.write("Filters and charts will go here…")
