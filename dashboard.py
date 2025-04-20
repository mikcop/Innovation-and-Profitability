import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Corporate R&D Investors Dashboard", layout="wide")
px.defaults.template = "plotly_white"

@st.cache_data(show_spinner="Loading data …")
def load_data():
    df = pd.read_csv("panel_2015_2018.csv")
    numeric_cols = ["rd", "ns", "capex", "op", "emp", "patCN", "patEP", "patJP", "patKR", "patUS", "TMnEU", "TMnUS"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

st.sidebar.title("Filter panel")
raw_df = load_data()

years = sorted(raw_df["year"].unique())
sel_year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

countries = sorted(raw_df["ctry_code"].dropna().unique())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)

sectors = sorted(raw_df["nace2"].dropna().unique())
sel_sectors = st.sidebar.multiselect("Sector (NACE2)", sectors, default=sectors)

mask = (
    (raw_df["year"] == sel_year)
    & raw_df["ctry_code"].isin(sel_countries)
    & raw_df["nace2"].isin(sel_sectors)
)
df = raw_df.loc[mask].copy()

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

with np.errstate(divide="ignore", invalid="ignore"):
    df["rd_intensity"] = df["rd"] / df["ns"]
    df["op_margin"] = df["op"] / df["ns"]

df = df.replace([np.inf, -np.inf], np.nan)

pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in df.columns]
if pat_cols:
    df["ip5_total"] = df[pat_cols].sum(axis=1)

tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in df.columns]
if tm_cols:
    df["tm_total"] = df[tm_cols].sum(axis=1)

kpi_rd = df["rd"].sum()
kpi_ns = df["ns"].sum()
kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns else np.nan
kpi_op_margin = df["op"].sum() / kpi_ns if kpi_ns else np.nan
pat_ip5 = df["ip5_total"].sum() if "ip5_total" in df else np.nan
tm_total = df["tm_total"].sum() if "tm_total" in df else np.nan

num_kpis = 6 if not np.isnan(pat_ip5) else 4
kpi_cols = st.columns(num_kpis)

kpi_cols[0].metric("Total R&D (€ M)", f"{kpi_rd:,.0f}")
kpi_cols[1].metric("Total Net Sales (€ M)", f"{kpi_ns:,.0f}")
kpi_cols[2].metric("R&D Intensity", f"{kpi_rd_intensity:.2%}")
kpi_cols[3].metric("Operating Margin", f"{kpi_op_margin:.2%}")

if num_kpis == 6:
    kpi_cols[4].metric("IP5 Patent Apps", f"{pat_ip5:,.0f}")
    kpi_cols[5].metric("EU/US TM Apps", f"{tm_total:,.0f}")

st.divider()

overview_tab, sector_tab, ip_tab, growth_tab, summary_tab = st.tabs([
    "Overview", "Sector Dive", "IP vs Financials", "Growth 2015‑18", "Global Insights"
])

with summary_tab:
    st.subheader("📘 Global Insights from JRC–OECD Report")

    st.markdown("""
    - **Top 2,000 R&D investors** account for **75% of global ICT patents** and **60% of ICT design rights**.
    - They operate in **>100 countries** and across an average of **9 sectors**.
    - **IP strategy adoption**:
        - >50% use full IP bundles (patents, trademarks, designs).
        - **Pharma** leads in R&D intensity per patent; **Machinery & Electronics** in output volume.
        - **USPTO** is the top patenting office for ICT; followed by **EPO** and **SIPO**.
        - **Design rights** are critical in ICT and Transport for product differentiation.
    - **Regional trends**:
        - **Asia (KR, CN)** shows strong ICT patent specialization.
        - **EU/US** more balanced in health, green tech, energy.
    """)

    fig = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=75,
            delta={"reference": 100, "valueformat": ".0f", "suffix": "%"},
            title={"text": "Share of Global ICT Patents by Top 2,000 R&D Firms"},
            number={"suffix": "%"}
        )
    )
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

    pie_data = pd.DataFrame({
        "Region": ["Asia ICT", "EU & US Diversified"],
        "Focus Share": [58, 42]
    })
    pie_chart = px.pie(pie_data, names="Region", values="Focus Share", title="Regional Innovation Focus")
    st.plotly_chart(pie_chart, use_container_width=True)

    st.info("This summary is derived from the EU JRC/OECD report: *World Corporate Top R&D Investors: Industrial Property Strategies in the Digital Economy* (2021 edition).")
