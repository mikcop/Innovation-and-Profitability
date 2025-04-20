import streamlit as st
import pandas as pd
import plotly.express as px

# Load data (replace with actual data path or connector)
df = pd.read_csv("panel_2015_2018.csv")  # Replace with actual path if needed

# Data preprocessing
df['rd_intensity'] = df['rd'] / df['ns']
df['profit_margin'] = df['op'] / df['ns']

# Sidebar filters
st.sidebar.title("Filter Data")
selected_year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()))
selected_country = st.sidebar.multiselect("Select Country", df['ctry_code'].unique(), default=df['ctry_code'].unique())
selected_sector = st.sidebar.multiselect("Select Sector (ISIC4)", df['isic4'].unique(), default=df['isic4'].unique())
rd_range = st.sidebar.slider("R&D Intensity Range", 0.0, 10.0, (0.0, 2.0))
profit_range = st.sidebar.slider("Profit Margin Range", -2.0, 2.0, (-1.0, 1.0))

# Filter data
filtered_df = df[(df['year'] == selected_year) &
                 (df['ctry_code'].isin(selected_country)) &
                 (df['isic4'].isin(selected_sector)) &
                 (df['rd_intensity'].between(*rd_range)) &
                 (df['profit_margin'].between(*profit_range))]

st.title("Corporate R&D Innovation Dashboard")

# Time-Series (full sample)
st.subheader("R&D and Financial Metrics Over Time")
time_series = df.groupby('year')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
fig_line = px.line(time_series, x='year', y=['rd', 'ns', 'capex', 'op'], markers=True,
                   labels={'value': 'Million €', 'variable': 'Metric'})
st.plotly_chart(fig_line)

# Bar Chart: Country-wise R&D
st.subheader("Country Comparison: R&D Investment")
country_rd = filtered_df.groupby('ctry_code')['rd'].sum().reset_index()
fig_bar = px.bar(country_rd, x='ctry_code', y='rd', color='rd', labels={'rd': 'R&D (€M)'})
st.plotly_chart(fig_bar)

# Scatter Plot: R&D Intensity vs Profit Margin
st.subheader("R&D Intensity vs Profit Margin")
fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code')
st.plotly_chart(fig_scatter)

# Histogram: Patent & Trademark Counts
st.subheader("Patent and Trademark Histogram")
fig_hist = px.histogram(filtered_df, x='patEP', nbins=30, title='European Patents')
st.plotly_chart(fig_hist)

# Table of filtered firms
st.subheader("Filtered Company-Level Data")
st.dataframe(filtered_df[['company_name', 'ctry_code', 'isic4', 'rd', 'ns', 'op', 'rd_intensity', 'profit_margin']])

# Download link
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data", data=csv, file_name='filtered_rd_data.csv', mime='text/csv')
