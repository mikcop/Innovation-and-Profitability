import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("panel_2015_2018.csv")

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

st.markdown("""
### 📘 Variable Legend (Legenda Variabili)
- **rd**: R&D investment (in million €) / Investimenti in R&S (in milioni di euro)
- **ns**: Net sales (in million €) / Fatturato netto (in milioni di euro)
- **capex**: Capital expenditures (in million €) / Spese in conto capitale
- **op**: Operating profits (in million €) / Profitti operativi
- **emp**: Number of employees / Dipendenti
- **rd_intensity**: R&D as a % of net sales / Intensità di R&S (% su vendite)
- **profit_margin**: Operating profits as a % of net sales / Margine operativo (% su vendite)
- **ctry_code**: Country code (ISO) / Codice paese
- **isic4**: ISIC Rev.4 sector classification / Settore industriale (ISIC Rev.4)
- **patEP**: Patent counts (European Patent Office) / Brevetti EPO
""")

# Time-Series (full sample)
st.subheader("R&D and Financial Metrics Over Time")
time_series = df.groupby('year')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
fig_line = px.line(time_series, x='year', y=['rd', 'ns', 'capex', 'op'], markers=True,
                   labels={'value': 'Million €', 'variable': 'Metric'})
st.plotly_chart(fig_line)

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(filtered_df[['rd', 'ns', 'capex', 'op', 'emp', 'rd_intensity', 'profit_margin']].describe())

# Correlation Heatmap
st.subheader("Correlation Matrix")
corr = filtered_df[['rd', 'ns', 'capex', 'op', 'emp', 'rd_intensity', 'profit_margin']].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', zmin=-1, zmax=1)
st.plotly_chart(fig_corr)

# R&D Intensity by Sector
st.subheader("Average R&D Intensity by Sector")
sector_group = df[df['year'] == selected_year].groupby('isic4')['rd_intensity'].mean().reset_index()
fig_sector = px.bar(sector_group, x='isic4', y='rd_intensity', labels={'rd_intensity': 'R&D Intensity'})
st.plotly_chart(fig_sector)

# Bar Chart: Country-wise R&D
st.subheader("Country Comparison: R&D Investment")
country_rd = filtered_df.groupby('ctry_code')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
fig_bar_ctry = px.bar(country_rd, x='ctry_code', y=['rd', 'ns', 'capex', 'op'], barmode='group',
                      labels={'value': '€M', 'variable': 'Metric', 'ctry_code': 'Country'})
st.plotly_chart(fig_bar_ctry)

# Company Comparison
st.subheader("Company Comparison: R&D vs Performance")
company_comp = filtered_df.groupby('company_name')[['rd', 'ns', 'op', 'rd_intensity', 'profit_margin']].mean().reset_index()
top_companies = company_comp.sort_values(by='rd', ascending=False).head(20)
fig_company = px.bar(top_companies, x='company_name', y='rd', color='profit_margin',
                     labels={'rd': 'R&D (€M)', 'company_name': 'Company', 'profit_margin': 'Profit Margin'},
                     title='Top 20 R&D Investors and their Profit Margins')
fig_company.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_company)

# Scatter Plot: R&D Intensity vs Profit Margin
st.subheader("R&D Intensity vs Profit Margin")
fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code')
st.plotly_chart(fig_scatter)

# Histogram: Patent & Trademark Counts
st.subheader("Patent and Trademark Histogram")
fig_hist = px.histogram(filtered_df, x='patEP', nbins=30, title='European Patents')
st.plotly_chart(fig_hist)

# Top R&D Firms Table
st.subheader("Top 10 Firms by R&D Investment")
top_rd = filtered_df.sort_values(by='rd', ascending=False).head(10)
st.dataframe(top_rd[['company_name', 'rd', 'ns', 'op', 'rd_intensity', 'profit_margin']])

# Filtered Company-Level Table
st.subheader("Filtered Company-Level Data")
st.dataframe(filtered_df[['company_name', 'ctry_code', 'isic4', 'rd', 'ns', 'op', 'rd_intensity', 'profit_margin']])

st.subheader("🔍 Strategic Key Insights")
st.markdown("""
- **Top R&D Investors** such as Alphabet, Samsung, and Microsoft maintain high R&D intensity paired with strong profit margins, showing mature innovation-to-market cycles.
- **Biotech & Pharma** sectors (e.g., TANVEX, OBSEVA) often show very high R&D intensity but negative profit margins due to long development horizons.
- **Country Highlights**:
  - **US**: Highest R&D spend and patenting activity, especially in IT and pharma.
  - **Germany & Japan**: Consistent investment in automotive and manufacturing sectors.
  - **South Korea & China**: Rapid rise in R&D intensity in electronics.
- **Correlation Trends**:
  - R&D intensity does **not linearly predict profitability**; efficient R&D management and sector context are critical.
  - Larger firms (by sales and headcount) tend to show stronger performance correlations.
- **Strategy Tip**: Firms should monitor R&D intensity to ensure it's aligned with expected market returns, and invest in IP management to turn innovation into competitive advantage.
""")

# Download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data", data=csv, file_name='filtered_rd_data.csv', mime='text/csv')
