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

# Derived innovation efficiency metrics
df['patents_per_rd'] = df['patEP'] / df['rd']
df['revenue_per_employee'] = df['ns'] / df['emp']

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

# Scatter: R&D Intensity vs Profit Margin (with Polynomial Fit)
# Show outliers on this chart
st.subheader("📌 R&D Intensity vs Profit Margin (Inverted-U Trend with Outliers)")

fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code',
                         title="All Companies")

# Polynomial fit
fit = np.polyfit(filtered_df['rd_intensity'], filtered_df['profit_margin'], 2)
fit_fn = np.poly1d(fit)

rd_range_vals = np.linspace(filtered_df['rd_intensity'].min(), filtered_df['rd_intensity'].max(), 200)
profit_pred = fit_fn(rd_range_vals)
fig_scatter.add_scatter(x=rd_range_vals, y=profit_pred, mode='lines', name='Quadratic Fit')

# Highlight underperformers
if not underperformers.empty:
    fig_scatter.add_scatter(
        x=underperformers['rd_intensity'],
        y=underperformers['profit_margin'],
        mode='markers',
        name='Underperformers',
        marker=dict(size=10, color='red', symbol='x'),
        text=underperformers['company_name']
    )
st.plotly_chart(fig_scatter)
st.subheader("📌 R&D Intensity vs Profit Margin (Inverted-U Trend)")

fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code')

# Polynomial fit
fit = np.polyfit(filtered_df['rd_intensity'], filtered_df['profit_margin'], 2)
fit_fn = np.poly1d(fit)

rd_range_vals = np.linspace(filtered_df['rd_intensity'].min(), filtered_df['rd_intensity'].max(), 200)
profit_pred = fit_fn(rd_range_vals)

fig_scatter.add_scatter(x=rd_range_vals, y=profit_pred, mode='lines', name='Quadratic Fit')
st.plotly_chart(fig_scatter)
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

    st.markdown("### 📊 KPI Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average R&D Intensity", f"{comp_df['rd_intensity'].mean():.2%}")
    col2.metric("Average Profit Margin", f"{comp_df['profit_margin'].mean():.2%}")
    col3.metric("Top R&D Investor", comp_df.loc[comp_df['rd'].idxmax(), 'company_name'])

    st.markdown("### 📉 Trend Over Time")
    trend_df = df[df['company_name'].isin(selected_companies)]
    trend_fig = px.line(trend_df, x='year', y='rd', color='company_name', markers=True,
                        labels={'rd': 'R&D (€M)', 'year': 'Year'})
    st.plotly_chart(trend_fig)
else:
    st.info("Select companies from the dropdown to compare their R&D and profitability.")
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

# Patent Efficiency Analysis
st.subheader("📚 Innovation Efficiency Metrics")

highlight_df = filtered_df.copy()
highlight_df['patents_per_rd'] = highlight_df['patEP'] / highlight_df['rd']
highlight_df['revenue_per_employee'] = highlight_df['ns'] / highlight_df['emp']

st.markdown("### 🧪 Top Efficient Innovators")
efficient = highlight_df.sort_values(by='patents_per_rd', ascending=False).head(10)
st.dataframe(efficient[['company_name', 'ctry_code', 'rd', 'patEP', 'patents_per_rd', 'revenue_per_employee']])

# Performance Outlier Filter
st.sidebar.subheader("Outlier Display Options")
view_mode = st.sidebar.radio("Select outlier type to display:", ["Underperformers", "Outperformers", "Both"])

# Calculate underperformers and outperformers
underperformers = highlight_df[(highlight_df['rd_intensity'] > highlight_df['rd_intensity'].quantile(rd_q)) &
                                (highlight_df['profit_margin'] < highlight_df['profit_margin'].quantile(profit_q))]
outperformers = highlight_df[(highlight_df['rd_intensity'] < highlight_df['rd_intensity'].quantile(1 - rd_q)) &
                              (highlight_df['profit_margin'] > highlight_df['profit_margin'].quantile(1 - profit_q))]
st.sidebar.markdown("---")
st.sidebar.subheader("Outlier Filter Settings")
rd_q = st.sidebar.slider("R&D Intensity Quantile Threshold", 0.5, 1.0, 0.75)
profit_q = st.sidebar.slider("Profit Margin Quantile Threshold", 0.0, 0.5, 0.25)

# Performance Outlier Filter
st.subheader("⚠️ Underperformers: High R&D, Low Profit")
underperformers = highlight_df[(highlight_df['rd_intensity'] > highlight_df['rd_intensity'].quantile(rd_q)) &
                                (highlight_df['profit_margin'] < highlight_df['profit_margin'].quantile(profit_q))] > highlight_df['rd_intensity'].quantile(0.75)) &
                                (highlight_df['profit_margin'] < highlight_df['profit_margin'].quantile(0.25))]
if not underperformers.empty:
    st.markdown("### 🚨 High R&D Spend, Low Profit Margin")
    st.dataframe(underperformers[['company_name', 'ctry_code', 'rd', 'rd_intensity', 'profit_margin']])
else:
    st.info("No significant underperformers found with current filters.")

# Show selected outliers
if view_mode in ["Underperformers", "Both"]:
    st.subheader("⚠️ Underperformers: High R&D, Low Profit")
    if not underperformers.empty:
        st.dataframe(underperformers[['company_name', 'ctry_code', 'rd', 'rd_intensity', 'profit_margin']])
    else:
        st.info("No significant underperformers found with current filters.")

if view_mode in ["Outperformers", "Both"]:
    st.subheader("🏆 Outperformers: Low R&D, High Profit")
    if not outperformers.empty:
        st.dataframe(outperformers[['company_name', 'ctry_code', 'rd', 'rd_intensity', 'profit_margin']])
    else:
        st.info("No significant outperformers found with current filters.")

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
