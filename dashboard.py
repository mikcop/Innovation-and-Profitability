import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Configure page
st.set_page_config(page_title="Company Panel Dashboard", layout="wide")

@st.cache_data
# Load and cache data
def load_data():
    df = pd.read_csv('panel_2015_2018.csv')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
# Year slider
years = sorted(df['year'].unique())
selected_year = st.sidebar.slider(
    "Select Year",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=int(max(years))
)

# Country selector
countries = sorted(df['ctry_code'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

# World rank range slider
min_wr, max_wr = int(df['worldrank'].min()), int(df['worldrank'].max())
selected_wr = st.sidebar.slider(
    "Select World Rank Range",
    min_value=min_wr,
    max_value=max_wr,
    value=(min_wr, max_wr)
)

# Metric selection
metrics = ['rd', 'ns', 'capex', 'op', 'emp']
metric = st.sidebar.selectbox(
    "Select Metric",
    options=metrics,
    index=0
)

# Apply filters
filtered_df = df[
    (df['year'] == selected_year) &
    (df['ctry_code'].isin(selected_countries)) &
    (df['worldrank'] >= selected_wr[0]) &
    (df['worldrank'] <= selected_wr[1])
]

# Main dashboard title
st.title(f"Company Panel Dashboard — {selected_year}")

# Top 10 Companies by Metric
st.subheader(f"Top 10 Companies by {metric.upper()}")
# Ensure filtered_df sorted by selected metric then worldrank
top10 = filtered_df.nlargest(10, metric)
fig1 = px.bar(
    top10,
    x='company_name',
    y=metric,
    color='ctry_code',
    labels={metric: metric.upper(), 'company_name': 'Company'},
    title=None
)
st.plotly_chart(fig1, use_container_width=True)

# Scatter: R&D vs Net Sales
st.subheader("R&D vs Net Sales")
fig2 = px.scatter(
    filtered_df,
    x='rd',
    y='ns',
    color='ctry_code',
    size='emp',
    hover_data=['company_name', 'worldrank'],
    labels={'rd': 'R&D Expenditure', 'ns': 'Net Sales', 'emp': 'Employees'},
    title=None
)
st.plotly_chart(fig2, use_container_width=True)

# Patent Distribution Heatmap for Selected Company
st.subheader("Patent Distribution Over Years")
company_list = sorted(df['company_name'].unique())
selected_company = st.selectbox("Select Company", company_list)
company_df = df[df['company_name'] == selected_company]
patent_cols = ['patCN', 'patEP', 'patJP', 'patKR', 'patUS']

# Prepare heatmap data
hm_df = company_df.set_index('year')[patent_cols]
fig3, ax = plt.subplots()
sns.heatmap(
    hm_df,
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    cmap='Blues',
    ax=ax
)
ax.set_ylabel("Year")
ax.set_xlabel("Patent Region")
ax.set_title(f"Patents for {selected_company}")
st.pyplot(fig3)

# Correlation Matrix
st.subheader("Correlation Matrix of Metrics")
corr = filtered_df.select_dtypes(include=np.number).corr()
fig4, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cmap='coolwarm',
    ax=ax2
)
ax2.set_title("Correlation Matrix")
st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("*Dashboard powered by Streamlit | Data: Panel 2015-2018* ")
