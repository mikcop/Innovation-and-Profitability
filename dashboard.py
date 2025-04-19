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
# 1. Calculate quartiles
rd_q  = filtered["rd_intensity"].quantile(0.75)
prof_q = filtered["profit_margin"].quantile(0.25)

# 2. Filter high-R&D but low-profit firms
highlight_df = filtered[
    (filtered["rd_intensity"] > rd_q) &
    (filtered["profit_margin"]  < prof_q)
]

# 3. Display if any exist
if not highlight_df.empty:
    st.markdown("### 🚩 High R&D, Low Profit Margin")
    st.dataframe(
        highlight_df[["company_name","ctry_code","rd_intensity","profit_margin"]]
    )
# Aggregate R&D spend by country
map_df = filtered.groupby("ctry_code")["rd"].sum().reset_index()

fig_map = px.choropleth(
    map_df,
    locations="ctry_code",
    color="rd",
    color_continuous_scale="Blues",
    projection="natural earth",
    labels={"rd":"R&D (€ M)"},
)
st.subheader("🌍 R&D Spend by Country (Map)")
st.plotly_chart(fig_map, use_container_width=True)
