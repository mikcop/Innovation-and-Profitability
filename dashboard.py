import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. Load & preprocess
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # … your preprocessing …
    return df

df = load_data("panel_2015_2018.csv")

# 2. Sidebar filters  
st.sidebar.title("🔍 Filter Data")
year       = st.sidebar.selectbox("Year", sorted(df.year.unique()))
# … other filters …
mask = (
    (df.year == year)
    # & your other conditions …
)
filtered = df.loc[mask]

# 3. Outlier table: MUST come after `filtered` is defined
if not filtered.empty:
    # compute quartiles
    rd_q   = filtered["rd_intensity"].quantile(0.75)
    prof_q = filtered["profit_margin"].quantile(0.25)

    # pick the high-R&D, low-profit firms
    highlight_df = filtered[
        (filtered["rd_intensity"] > rd_q) &
        (filtered["profit_margin"]  < prof_q)
    ]

    if not highlight_df.empty:
        st.markdown("### 🚩 High R&D, Low Profit Margin")
        st.dataframe(
            highlight_df[["company_name","ctry_code","rd_intensity","profit_margin"]]
        )

# 4. Your charts (time series, scatter, etc.) go here…
