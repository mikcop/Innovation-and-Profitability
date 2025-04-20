import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Corporate R&D Investors Dashboard",
    layout="wide",
)

# --------------------------------------------------
# 1. DATA LOADING & PREPARATION
# --------------------------------------------------

@st.cache_data(show_spinner="Loading data …")
def load_data():
    """Read merged panel dataset.

    Expected column names (case‑sensitive):
        company_id, company_name, ctry_code, worldrank,
        nace2, isic4, year, rd, ns, capex, op, emp,
        patCN, patEP, patJP, patKR, patUS, TMnEU, TMnUS
    """
    df = pd.read_csv("panel_2015_2018.csv")

    numeric_cols = [
        "rd", "ns", "capex", "op", "emp",
        "patCN", "patEP", "patJP", "patKR", "patUS",
        "TMnEU", "TMnUS"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

# --------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# --------------------------------------------------

st.sidebar.title("Filters")
df = load_data()

years = sorted(df["year"].unique())
sel_year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

countries = sorted(df["ctry_code"].dropna().unique())
sel_countries = st.sidebar.multiselect(
    "Country", options=countries, default=countries, placeholder="Select country codes"
)

sectors = sorted(df["nace2"].dropna().unique())
sel_sectors = st.sidebar.multiselect(
    "Sector (NACE2)", options=sectors, default=sectors, placeholder="Select sectors"
)

# --------------------------------------------------
# 3. FILTERING
# --------------------------------------------------

mask = (
    (df["year"] == sel_year)
    & (df["ctry_code"].isin(sel_countries))
    & (df["nace2"].isin(sel_sectors))
)
filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --------------------------------------------------
# 4. DERIVED METRICS
# --------------------------------------------------

filtered["rd_intensity"] = filtered["rd"] / filtered["ns"]
filtered["op_margin"] = filtered["op"] / filtered["ns"]

kpi_rd = filtered["rd"].sum()
kpi_ns = filtered["ns"].sum()
kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns else 0
kpi_op_margin = filtered["op"].sum() / kpi_ns if kpi_ns else 0

pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in filtered.columns]
pat_ip5 = filtered[pat_cols].fillna(0).sum().sum() if pat_cols else np.nan

tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in filtered.columns]
tm_total = filtered[tm_cols].fillna(0).sum().sum() if tm_cols else np.nan

# --------------------------------------------------
# 5. KPI CARDS
# --------------------------------------------------

num_kpis = 6 if not np.isnan(pat_ip5) else 4
cols = st.columns(num_kpis)

cols[0].metric("Total R&D (€ M)", f"{kpi_rd:,.0f}")
cols[1].metric("Total Net Sales (€ M)", f"{kpi_ns:,.0f}")
cols[2].metric("R&D Intensity", f"{kpi_rd_intensity:.2%}")
cols[3].metric("Operating Margin", f"{kpi_op_margin:.2%}")

if num_kpis == 6:
    cols[4].metric("IP5 Patent Apps", f"{pat_ip5:,.0f}")
    cols[5].metric("EU/US Trademarks", f"{tm_total:,.0f}")

st.divider()

# --------------------------------------------------
# 6. TOP INVESTORS BAR CHART
# --------------------------------------------------

st.subheader("Top 10 R&D Investors")

top10 = filtered.nlargest(10, "rd")
fig_bar = px.bar(
    top10,
    x="company_name",
    y="rd",
    text="rd",
    labels={"rd": "R&D (€ Million)", "company_name": "Company"},
)
fig_bar.update_layout(xaxis_tickangle=-35, yaxis_title="R&D (€ M)")
st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------------------------------
# 7. BUBBLE CHART – INTENSITY VS MARGIN
# --------------------------------------------------

st.subheader("R&D Intensity vs Operating Margin")

fig_scatter = px.scatter(
    filtered,
    x="rd_intensity",
    y="op_margin",
    size="ns",
    color="nace2",
    hover_name="company_name",
    labels={
        "rd_intensity": "R&D Intensity (R&D / Net Sales)",
        "op_margin": "Operating Margin (OP / Net Sales)",
        "ns": "Net Sales (€ M)",
        "nace2": "Sector (NACE2)",
    },
)
fig_scatter.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------
# 8. TREND LINES OVER TIME
# --------------------------------------------------

st.subheader("R&D and Net Sales Trend (2015‑2018)")

trend_df = (
    df[(df["ctry_code"].isin(sel_countries)) & (df["nace2"].isin(sel_sectors))]
    .groupby("year")[["rd", "ns"]]
    .sum()
    .reset_index()
)

fig_trend = px.line(
    trend_df,
    x="year",
    y=["rd", "ns"],
    markers=True,
    labels={"value": "Amount (€ Million)", "variable": "Metric"},
)
st.plotly_chart(fig_trend, use_container_width=True)

# --------------------------------------------------
# 9. DATA TABLE & DOWNLOAD
# --------------------------------------------------

with st.expander("Show filtered data"):
    st.dataframe(filtered, use_container_width=True)

@st.cache_data
def convert_df(_df):
    return _df.to_csv(index=False).encode("utf-8")

download_btn = st.download_button(
    "📥 Download filtered CSV",
    convert_df(filtered),
    file_name="filtered_rd_dataset.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.caption(
    "Data source: EC‑JRC / OECD COR&DIP © 2021 – *Top 2 000 Corporate R&D Investors*"
)
