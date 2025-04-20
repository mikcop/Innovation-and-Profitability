import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("panel_2015_2018.csv", sheet_name="Dataset")
    return df

df = load_data()

# Title
st.title("Company Innovation & Financial Dashboard (2015–2018)")

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.multiselect("Select Country", options=sorted(df['ctry_code'].dropna().unique()), default=['US'])
selected_years = st.sidebar.multiselect("Select Year", options=sorted(df['year'].dropna().unique()), default=[2018])
selected_ranks = st.sidebar.slider("Select World Rank Range", int(df['worldrank'].min()), int(df['worldrank'].max()), (1, 100))

filtered_df = df[
    df['ctry_code'].isin(selected_country) &
    df['year'].isin(selected_years) &
    df['worldrank'].between(selected_ranks[0], selected_ranks[1])
]

# Show filtered data
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(20))

# KPI Summary
st.subheader("Key Performance Indicators")
kpi1 = filtered_df['rd'].sum()
kpi2 = filtered_df['ns'].sum()
kpi3 = filtered_df['emp'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total R&D Investment", f"${kpi1:,.0f}")
col2.metric("Total Net Sales", f"${kpi2:,.0f}")
col3.metric("Total Employment", f"{kpi3:,.0f} Employees")

# Line chart: R&D over time
st.subheader("R&D Investment Over Time")
rd_time = df[df['ctry_code'].isin(selected_country)].groupby(['year', 'ctry_code'])['rd'].sum().unstack()
st.line_chart(rd_time)

# Bar chart: Top 10 companies by R&D
st.subheader("Top 10 Companies by R&D Investment")
top_companies = filtered_df.groupby('company_name')['rd'].sum().nlargest(10)
fig, ax = plt.subplots()
top_companies.plot(kind='barh', ax=ax)
ax.set_xlabel("R&D Investment")
ax.set_ylabel("Company")
st.pyplot(fig)

# Patent summary table
st.subheader("Patent Summary")
patents = filtered_df[['company_name', 'patCN', 'patEP', 'patJP', 'patKR', 'patUS']].dropna()
patents['Total Patents'] = patents[['patCN', 'patEP', 'patJP', 'patKR', 'patUS']].sum(axis=1)
patents = patents.sort_values(by='Total Patents', ascending=False).head(10)
st.dataframe(patents)
