import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load data and filter for top 1000 firms by global rank
full_df = pd.read_csv("panel_2015_2018.csv")


# Data preprocessing
df['rd_intensity'] = df['rd'] / df['ns']
df['profit_margin'] = df['op'] / df['ns']
df['patents_per_rd'] = df['patEP'] / df['rd']
df['revenue_per_employee'] = df['ns'] / df['emp']

# Sidebar filters
st.sidebar.title("🔍 Filter Data")
selected_year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()))
selected_country = st.sidebar.multiselect("Select Country", df['ctry_code'].unique(), default=df['ctry_code'].unique())
selected_sector = st.sidebar.multiselect("Select Sector (ISIC4)", df['isic4'].unique(), default=df['isic4'].unique())
rd_range = st.sidebar.slider("R&D Intensity Range", 0.0, 10.0, (0.0, 2.0))
profit_range = st.sidebar.slider("Profit Margin Range", -2.0, 2.0, (-1.0, 1.0))
rank_range = st.sidebar.slider("Select Global Rank Range", 1, 8000, (1, 100))

# Filter data
filtered_df = df[
    (df['year'] == selected_year) &
    (df['ctry_code'].isin(selected_country)) &
    (df['isic4'].isin(selected_sector)) &
    (df['rd_intensity'].between(*rd_range)) &
    (df['profit_margin'].between(*profit_range)) &
    (df['worldrank'].between(*rank_range))
]

st.title("🚀 Corporate R&D Innovation Dashboard")

# Quick Metrics
st.subheader("📊 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Avg R&D Intensity", f"{filtered_df['rd_intensity'].mean():.2%}")
col2.metric("Avg Profit Margin", f"{filtered_df['profit_margin'].mean():.2%}")
col3.metric("Top R&D Firm", filtered_df.loc[filtered_df['rd'].idxmax(), 'company_name'])

# Variable legend
with st.expander("📘 Variable Legend (Legenda Variabili)"):
    st.markdown("""
    - **rd**: R&D investment (in million €)
    - **ns**: Net sales
    - **capex**: Capital expenditures
    - **op**: Operating profits
    - **emp**: Number of employees
    - **rd_intensity**: R&D / Sales
    - **profit_margin**: Operating Profit / Sales
    - **ctry_code**: Country code
    - **isic4**: ISIC sector code
    - **patEP**: European patents
    """)

# Time-Series Chart
time_series = df.groupby('year')[['rd', 'ns', 'capex', 'op']].sum().reset_index()
for col in ['rd', 'ns', 'capex', 'op']:
    time_series[f'{col}_growth'] = time_series[col].pct_change() * 100

fig_line = px.line(time_series, x='year', y=['rd', 'ns', 'capex', 'op'], markers=True,
                   labels={'value': '€M', 'variable': 'Metric'})
st.subheader("📈 R&D and Financial Metrics Over Time")
st.plotly_chart(fig_line)

st.markdown("### 📊 Year-on-Year Growth Rates")
st.dataframe(time_series[['year', 'rd_growth', 'ns_growth', 'capex_growth', 'op_growth']])

# Scatterplot: R&D Intensity vs Profit Margin
fig_scatter = px.scatter(filtered_df, x='rd_intensity', y='profit_margin',
                         hover_data=['company_name'], color='ctry_code',
                         title="R&D Intensity vs Profit Margin")
fit = np.polyfit(filtered_df['rd_intensity'], filtered_df['profit_margin'], 2)
fit_fn = np.poly1d(fit)
rd_range_vals = np.linspace(filtered_df['rd_intensity'].min(), filtered_df['rd_intensity'].max(), 200)
fig_scatter.add_scatter(x=rd_range_vals, y=fit_fn(rd_range_vals), mode='lines', name='Quadratic Fit')
st.plotly_chart(fig_scatter)

# Bubble Chart
st.subheader("🫧 Bubble Chart: R&D Intensity vs Profit Margin by Patent Output")
bubble_df = filtered_df.copy()
bubble_df['patEP_size'] = bubble_df['patEP'].fillna(0).clip(upper=500)
fig_bubble = px.scatter(bubble_df, x='rd_intensity', y='profit_margin', size='patEP_size',
                        color='ctry_code', hover_name='company_name', size_max=60,
                        title="Bubble Chart: R&D Intensity vs Profitability (Bubble = Patent Count)")
st.plotly_chart(fig_bubble)

# Bar Chart: R&D Intensity by Sector
sector_group = df[df['year'] == selected_year].groupby('isic4')['rd_intensity'].mean().reset_index()
sector_group = sector_group.sort_values(by='rd_intensity', ascending=False).head(20)
fig_sector = px.bar(sector_group, y='isic4', x='rd_intensity', orientation='h',
                    title='Top 20 Avg R&D Intensity by Sector',
                    text_auto='.2f', color='rd_intensity', color_continuous_scale='Blues')
fig_sector.update_layout(yaxis_title='ISIC Sector Code', xaxis_title='R&D Intensity (%)')
st.plotly_chart(fig_sector)

# Sector Heatmap
st.subheader("🌡️ Sector Heatmap: Innovation & Performance Metrics")
sector_metrics = df[df['year'] == selected_year].groupby('isic4')[
    ['rd_intensity', 'profit_margin', 'revenue_per_employee', 'patEP', 'emp']
].mean().round(2)
sector_metrics.columns = [
    'Avg R&D Intensity',
    'Avg Profit Margin',
    'Revenue per Employee',
    'Avg Patents (EP)',
    'Avg Employees'
]
fig_sector_heatmap = px.imshow(
    sector_metrics,
    text_auto=".2f",
    labels=dict(x="Metric", y="ISIC4 Sector", color="Value"),
    title="Heatmap: Sector Innovation & Financial Performance (Year {})".format(selected_year),
    aspect="auto",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_sector_heatmap)

# Country R&D Investment
st.subheader("🌍 R&D Investment by Country")
country_rd = filtered_df.groupby('ctry_code')['rd'].sum().reset_index().sort_values(by='rd', ascending=False).head(20)
fig_country = px.bar(country_rd, x='ctry_code', y='rd', color='rd', labels={'rd': 'R&D (€M)'})
st.plotly_chart(fig_country)

# R&D Rank Evolution
display_df = df[df['worldrank'].between(*rank_range)].copy()
display_df['rd_rank'] = display_df.groupby('year')['rd'].rank(ascending=False)
available_companies = display_df['company_name'].unique()
selected_companies = st.multiselect(
    "Select Companies to Track (max 10)",
    options=sorted(available_companies),
    default=sorted(display_df.groupby('company_name')['rd'].sum().nlargest(5).index),
    max_selections=10
)
rank_subset = display_df[display_df['company_name'].isin(selected_companies)]
fig_rank_line = px.line(rank_subset, x='year', y='rd_rank', color='company_name', markers=True,
                        title='R&D Rank Evolution (Lower = Better)')
fig_rank_line.update_yaxes(autorange='reversed')
st.plotly_chart(fig_rank_line)

# Top Movers 2015 vs 2018
st.subheader(f"📉 Top Movers in R&D Ranking (2015 vs 2018, Rank {rank_range[0]}–{rank_range[1]})")
rank_2015 = df[(df['year'] == 2015) & df['worldrank'].between(*rank_range)][['company_id', 'company_name', 'rd']].copy()
rank_2015['rank_2015'] = rank_2015['rd'].rank(ascending=False)
rank_2018 = df[(df['year'] == 2018) & df['worldrank'].between(*rank_range)][['company_id', 'company_name', 'rd']].copy()
rank_2018['rank_2018'] = rank_2018['rd'].rank(ascending=False)
ranking = pd.merge(rank_2015[['company_id', 'company_name', 'rank_2015']],
                   rank_2018[['company_id', 'rank_2018']], on='company_id')
ranking['rank_shift'] = ranking['rank_2015'] - ranking['rank_2018']
st.markdown("#### 🚀 Biggest Climbers")
st.dataframe(ranking.sort_values(by='rank_shift', ascending=False).head(10))
st.markdown("#### 📉 Biggest Decliners")
st.dataframe(ranking.sort_values(by='rank_shift').head(10))

# Innovation Efficiency
st.subheader("📚 Innovation Efficiency Metrics")
efficient = filtered_df.copy()
efficient['patents_per_rd'] = efficient['patEP'] / efficient['rd']
efficient['revenue_per_employee'] = efficient['ns'] / efficient['emp']
st.markdown("### 🧪 Top Efficient Innovators")
st.dataframe(efficient.sort_values(by='patents_per_rd', ascending=False).head(10)[
    ['company_name', 'ctry_code', 'rd', 'patEP', 'patents_per_rd', 'revenue_per_employee']])

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap: Innovation vs Financial Metrics")
corr_vars = ['rd', 'rd_intensity', 'patEP', 'ns', 'op', 'profit_margin', 'emp', 'revenue_per_employee']
corr_df = filtered_df[corr_vars].dropna()
corr_matrix = corr_df.corr()
fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r',
                        title="Correlation Matrix – Innovation & Financial Variables")
st.plotly_chart(fig_heatmap)

# Download filtered data
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Filtered Data", data=csv, file_name='filtered_rd_data.csv', mime='text/csv')

