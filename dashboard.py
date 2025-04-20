import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv("panel_2015_2018.csv")

# Clean and enrich data
df['total_patents'] = df[['patCN', 'patEP', 'patJP', 'patKR', 'patUS']].sum(axis=1)
df.dropna(subset=['rd', 'op'], inplace=True)

# Sidebar filters
st.sidebar.title("🔎 Filters")
year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()))
min_rank, max_rank = st.sidebar.slider("World Rank Range", 1, df['worldrank'].max(), (1, 100))
selected_country = st.sidebar.multiselect("Country", df['ctry_code'].unique(), default=list(df['ctry_code'].unique()))

# Apply filters
filtered_df = df[(df['year'] == year) &
                 (df['worldrank'].between(min_rank, max_rank)) &
                 (df['ctry_code'].isin(selected_country))]

st.title("📈 Innovation & Profitability Dashboard")
st.markdown("""
This dashboard explores whether innovation (R&D, patents) correlates with company profitability.
Use the sidebar to filter by year, country, and world rank.
""")

# Section 1: Overview
st.header("1. Key Innovation & Performance Metrics")
st.metric("Total R&D Investment", f"${filtered_df['rd'].sum():,.0f}")
st.metric("Total Operating Profit", f"${filtered_df['op'].sum():,.0f}")
st.metric("Total Patents", f"{int(filtered_df['total_patents'].sum())}")

# Section 2: Regression Analysis
st.header("2. Regression: R&D and Patents vs Profit")
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.regplot(data=filtered_df, x='rd', y='op', ax=axs[0])
axs[0].set_title('R&D vs Operating Profit')
sns.regplot(data=filtered_df, x='total_patents', y='op', ax=axs[1])
axs[1].set_title('Total Patents vs Operating Profit')
st.pyplot(fig)

# Correlation display
corr_val = filtered_df[['rd', 'total_patents', 'op']].corr()
st.subheader("Correlation Matrix")
st.dataframe(corr_val.style.background_gradient(cmap='coolwarm'))

# Section 3: Multivariate Analysis
st.header("3. Multivariate Exploration")
selected_vars = st.multiselect("Select Variables for Pairplot", ['rd', 'op', 'capex', 'ns', 'emp', 'total_patents'], default=['rd', 'op', 'total_patents'])
if len(selected_vars) > 1:
    sns.set(style="ticks")
    fig_pair = sns.pairplot(filtered_df[selected_vars].dropna())
    st.pyplot(fig_pair)

# Section 4: Firm-level Explorer
st.header("4. Company-Level Data Explorer")
st.dataframe(filtered_df[['company_name', 'ctry_code', 'worldrank', 'rd', 'op', 'total_patents']].sort_values(by='op', ascending=False))
