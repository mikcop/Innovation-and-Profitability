import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load full data
full_df = pd.read_csv("panel_2015_2018.csv")
df = full_df.copy()
df = full_df.groupby('year').head(1000).reset_index(drop=True).isin(sampled_ids)]

# Data preprocessing
df['rd_intensity'] = df['rd'] / df['ns']
df['profit_margin'] = df['op'] / df['ns']

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Worldrank Filter")
rank_min, rank_max = int(df['worldrank'].min()), int(df['worldrank'].max())
selected_rank = st.sidebar.slider("Worldrank Range (1 = Top R&D spender)", rank_min, rank_max, (rank_min, rank_max))
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
                 (df['profit_margin'].between(*profit_range)) &
                 (df['worldrank'].between(*selected_rank))].isin(selected_country)) &
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
st.markdown("""#### ✅ Assumption: Firms that invest in R&D consistently grow in scale
**Test:** Examine time-series of R&D, sales, CAPEX, and profit.
**Visual:** Line chart + growth rate table shows trends from 2015 to 2018.""")
st.subheader("📈 R&D and Financial Metrics Over Time")
time_series = df.groupby('year')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
time_series['rd_growth'] = time_series['rd'].pct_change() * 100
time_series['sales_growth'] = time_series['ns'].pct_change() * 100
time_series['capex_growth'] = time_series['capex'].pct_change() * 100
time_series['profit_growth'] = time_series['op'].pct_change() * 100

fig_line = px.line(time_series, x='year', y=['rd', 'ns', 'capex', 'op'], markers=True,
                   labels={'value': '€M', 'variable': 'Metric'})
st.plotly_chart(fig_line)

st.markdown("### 📊 Year-on-Year Growth Rates")
st.dataframe(time_series[['year', 'rd_growth', 'sales_growth', 'capex_growth', 'profit_growth']])

# Scatter: R&D Intensity vs Profit Margin (with Polynomial Fit)
st.markdown("""#### ✅ Assumption: R&D intensity improves profit, but only to a point
**Test:** Quadratic regression tests for inverted-U between R&D intensity and profit margin.
**Visual:** Scatterplot with fitted curve.""")

fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code',
                         title="All Companies")

# Polynomial fit
fit = np.polyfit(filtered_df['rd_intensity'], filtered_df['profit_margin'], 2)
fit_fn = np.poly1d(fit)

rd_range_vals = np.linspace(filtered_df['rd_intensity'].min(), filtered_df['rd_intensity'].max(), 200)
profit_pred = fit_fn(rd_range_vals)
fig_scatter.add_scatter(x=rd_range_vals, y=profit_pred, mode='lines', name='Quadratic Fit')

st.plotly_chart(fig_scatter)



# Bar: Sector-level R&D Intensity
st.markdown("""#### ✅ Assumption: R&D intensity differs strongly across industries
**Test:** Compare mean R&D intensity across ISIC sectors.
**Visual:** Top 10 bar chart with ISIC legend.""")
st.subheader("🏭 Top Avg R&D Intensity by Sector")
sector_group = df[df['year'] == selected_year].groupby('isic4')['rd_intensity'].mean().reset_index()
sector_group = sector_group.sort_values(by='rd_intensity', ascending=False).head(20)
fig_sector = px.bar(
    sector_group,
    y='isic4',
    x='rd_intensity',
    orientation='h',
    labels={'rd_intensity': 'R&D Intensity'},
    title='Top 20 Avg R&D Intensity by Sector',
    text_auto='.2f',
    color='rd_intensity',
    color_continuous_scale='Blues')
fig_sector.update_layout(
    yaxis_title='ISIC Sector Code',
    xaxis_title='R&D Intensity (%)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
    title_font_size=18
)',
    xaxis_tickangle=-45,
    plot_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
    title_font_size=18
)
st.plotly_chart(fig_sector)

with st.expander("🏷️ Sector Legend (ISIC4 to Description)"):
    st.markdown("""
    - **21**: Pharmaceuticals
    - **26**: Electronics & Optical
    - **29-30**: Automotive & Transport
    - **62-63**: IT & Software
    - **10-12**: Food, Beverages, Tobacco
    - **20**: Chemicals
    - **27**: Electrical Equipment
    - **58-60**: Publishing & Broadcasting
    - **28**: Machinery & Equipment
    - **61**: Telecommunications
    """)

# Country Comparison
st.markdown("""#### ✅ Assumption: High R&D by country aligns with innovation leadership
**Test:** Compare country R&D totals
**Visual:** Bar chart by country for selected year.""")
st.subheader("🌍 R&D Investment by Country")
country_rd = filtered_df.groupby('ctry_code')['rd'].sum().reset_index()
country_rd = country_rd.sort_values(by='rd', ascending=False).head(20)
fig_country = px.bar(country_rd, x='ctry_code', y='rd', color='rd', labels={'rd': 'R&D (€M)'})
st.plotly_chart(fig_country)

# Company R&D Rank Shifts
st.markdown("""#### ✅ Assumption: Global R&D leaders shift slowly, but top movers exist
**Test:** Rank changes from 2015 to 2018
**Visual:** Tables of biggest climbers and decliners.""")
st.subheader("📉 Top Movers in R&D Ranking (2015–2018)")
rank_2015 = df[df['year'] == 2015][['company_id', 'company_name', 'rd']].copy()
rank_2015['rank_2015'] = rank_2015['rd'].rank(ascending=False)

rank_2018 = df[df['year'] == 2018][['company_id', 'company_name', 'rd']].copy()
rank_2018['rank_2018'] = rank_2018['rd'].rank(ascending=False)

ranking = pd.merge(rank_2015[['company_id', 'company_name', 'rank_2015']],
                   rank_2018[['company_id', 'rank_2018']], on='company_id')
ranking['rank_shift'] = ranking['rank_2015'] - ranking['rank_2018']

movers_up = ranking.sort_values(by='rank_shift', ascending=False).head(10)
movers_down = ranking.sort_values(by='rank_shift').head(10)

st.markdown("#### 🚀 Biggest Climbers")
st.dataframe(movers_up[['company_name', 'rank_2015', 'rank_2018', 'rank_shift']])
st.markdown("#### 📉 Biggest Decliners")
st.dataframe(movers_down[['company_name', 'rank_2015', 'rank_2018', 'rank_shift']])

# Company Comparison


st.subheader("🏢 Top 10 R&D Firms")
top_rd = filtered_df.sort_values(by='rd', ascending=False).head(10)
fig_top = px.bar(top_rd, x='company_name', y='rd', color='profit_margin',
                 labels={'rd': 'R&D (€M)', 'company_name': 'Company'})
fig_top.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_top)

# Patent Efficiency Analysis
st.markdown("""#### ✅ Assumption: Efficient innovators generate more patents per euro
**Test:** Calculate `patEP / rd` and display top 10
**Visual:** Table showing patent efficiency and revenue per employee.""")
st.subheader("📚 Innovation Efficiency Metrics")

highlight_df = filtered_df.copy()
highlight_df['patents_per_rd'] = highlight_df['patEP'] / highlight_df['rd']
highlight_df['revenue_per_employee'] = highlight_df['ns'] / highlight_df['emp']

st.markdown("### 🧪 Top Efficient Innovators")
efficient = highlight_df.sort_values(by='patents_per_rd', ascending=False).head(10)
st.dataframe(efficient[['company_name', 'ctry_code', 'rd', 'patEP', 'patents_per_rd', 'revenue_per_employee']])


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
