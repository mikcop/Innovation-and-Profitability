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
st.sidebar.title("🔍 Filter Data")
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

st.title("🚀 Corporate R&D Innovation Dashboard")

# Quick Metrics
st.subheader("📊 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Avg R&D Intensity", f"{filtered_df['rd_intensity'].mean():.2%}")
col2.metric("Avg Profit Margin", f"{filtered_df['profit_margin'].mean():.2%}")
col3.metric("Top R&D Firm", filtered_df.loc[filtered_df['rd'].idxmax(), 'company_name'])

with st.expander("📘 Variable Legend (Legenda Variabili)"):
    st.markdown("""
    - **rd**: R&D investment (in million €) / Investimenti in R&S
    - **ns**: Net sales / Fatturato netto
    - **capex**: Capital expenditures / Spese in conto capitale
    - **op**: Operating profits / Profitti operativi
    - **emp**: Number of employees / Dipendenti
    - **rd_intensity**: R&D as % of sales / Intensità R&S
    - **profit_margin**: Operating profit % / Margine operativo
    - **ctry_code**: Country (ISO) / Codice paese
    - **isic4**: Sector classification / Settore ISIC
    - **patEP**: European patents / Brevetti EPO
    """)

# Time-Series Chart
st.subheader("📈 R&D and Financial Metrics Over Time")
time_series = df.groupby('year')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
fig_line = px.line(time_series, x='year', y=['rd', 'ns', 'capex', 'op'], markers=True,
                   labels={'value': '€M', 'variable': 'Metric'})
st.plotly_chart(fig_line)

# Scatter: R&D Intensity vs Profit Margin
st.subheader("📌 R&D Intensity vs Profit Margin")
fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code')
st.plotly_chart(fig_scatter)

# Bar: Sector-level R&D Intensity
st.subheader("🏭 Avg R&D Intensity by Sector")
sector_group = df[df['year'] == selected_year].groupby('isic4')['rd_intensity'].mean().reset_index()
fig_sector = px.bar(sector_group, x='isic4', y='rd_intensity', labels={'rd_intensity': 'R&D Intensity'})
st.plotly_chart(fig_sector)

# Country Comparison
st.subheader("🌍 R&D Investment by Country")
country_rd = filtered_df.groupby('ctry_code')['rd'].sum().reset_index()
fig_country = px.bar(country_rd, x='ctry_code', y='rd', color='rd', labels={'rd': 'R&D (€M)'})
st.plotly_chart(fig_country)

# Company Comparison
st.subheader("🏢 Compare Selected Companies")

company_options = filtered_df['company_name'].unique()
selected_companies = st.multiselect("Choose companies to compare", company_options)

if selected_companies:
    comp_df = filtered_df[filtered_df['company_name'].isin(selected_companies)]
    st.markdown("### 🔍 Company R&D vs Profit Comparison")
    fig_compare = px.bar(comp_df, x='company_name', y='rd', color='profit_margin',
                         hover_data=['rd_intensity'],
                         labels={'rd': 'R&D (€M)', 'company_name': 'Company', 'profit_margin': 'Profit Margin'})
    fig_compare.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_compare)

    st.markdown("### 📈 Performance Table")
    st.dataframe(comp_df[['company_name', 'ctry_code', 'isic4', 'rd', 'ns', 'op', 'rd_intensity', 'profit_margin']])
else:
    st.info("Select companies from the dropdown to compare their R&D and profitability.")
st.subheader("🏢 Top 10 R&D Firms")
top_rd = filtered_df.sort_values(by='rd', ascending=False).head(10)
fig_top = px.bar(top_rd, x='company_name', y='rd', color='profit_margin',
                 labels={'rd': 'R&D (€M)', 'company_name': 'Company'})
fig_top.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_top)

# Strategic Insights
st.subheader("🧠 Strategic Key Insights")
st.markdown("""
- **R&D Leaders**: Alphabet, Samsung, Microsoft link high R&D with strong profit margins.
- **Biotech Sector**: High R&D intensity but negative profit margin—long-term investment.
- **Top Countries**: US, KR, JP dominate R&D spend, especially in IT & pharma.
- **Balance Needed**: High R&D ≠ high profits. Efficiency and sector timing matter.
- **Tip**: Use this tool to benchmark innovation strategy vs. financial return.
""")

# Download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data", data=csv, file_name='filtered_rd_data.csv', mime='text/csv')
