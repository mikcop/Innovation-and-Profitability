import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLES
# ------------------------------------------------------------------

st.set_page_config(page_title="Corporate R&D Investors Dashboard", layout="wide")
px.defaults.template = "plotly_white"

# ------------------------------------------------------------------
# 1  DATA LOADING & PREPARATION
# ------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data …")
def load_data():
    df = pd.read_csv("panel_2015_2018.csv")

    numeric_cols = [
        "rd", "ns", "capex", "op", "emp",
        "patCN", "patEP", "patJP", "patKR", "patUS",
        "TMnEU", "TMnUS",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

# ------------------------------------------------------------------
# 2  SIDEBAR FILTERS
# ------------------------------------------------------------------

st.sidebar.title("Filter panel")
raw_df = load_data()

# Year selector (single year – can extend later)
years = sorted(raw_df["year"].unique())
sel_year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

# Country & sector multi‑selects
countries = sorted(raw_df["ctry_code"].dropna().unique())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)

sectors = sorted(raw_df["nace2"].dropna().unique())
sel_sectors = st.sidebar.multiselect("Sector (NACE2)", sectors, default=sectors)

# ------------------------------------------------------------------
# 3  FILTERING & CORE METRICS
# ------------------------------------------------------------------

mask = (
    (raw_df["year"] == sel_year)
    & raw_df["ctry_code"].isin(sel_countries)
    & raw_df["nace2"].isin(sel_sectors)
)
df = raw_df.loc[mask].copy()

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# Derived ratios
with np.errstate(divide="ignore", invalid="ignore"):
    df["rd_intensity"] = df["rd"] / df["ns"]
    df["op_margin"] = df["op"] / df["ns"]

df = df.replace([np.inf, -np.inf], np.nan)

# Patent / TM aggregations (if available)
pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in df.columns]
if pat_cols:
    df["ip5_total"] = df[pat_cols].sum(axis=1)

tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in df.columns]
if tm_cols:
    df["tm_total"] = df[tm_cols].sum(axis=1)

# KPI aggregates
kpi_rd = df["rd"].sum()
kpi_ns = df["ns"].sum()
kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns else np.nan
kpi_op_margin = df["op"].sum() / kpi_ns if kpi_ns else np.nan
pat_ip5 = df["ip5_total"].sum() if "ip5_total" in df else np.nan
tm_total = df["tm_total"].sum() if "tm_total" in df else np.nan

# ------------------------------------------------------------------
# 4  HEADER KPI CARDS
# ------------------------------------------------------------------

num_kpis = 6 if not np.isnan(pat_ip5) else 4
kpi_cols = st.columns(num_kpis)

kpi_cols[0].metric("Total R&D (\u20AC M)", f"{kpi_rd:,.0f}")
kpi_cols[1].metric("Total Net Sales (\u20AC M)", f"{kpi_ns:,.0f}")
kpi_cols[2].metric("R&D Intensity", f"{kpi_rd_intensity:.2%}")
kpi_cols[3].metric("Operating Margin", f"{kpi_op_margin:.2%}")

if num_kpis == 6:
    kpi_cols[4].metric("IP5 Patent Apps", f"{pat_ip5:,.0f}")
    kpi_cols[5].metric("EU/US TM Apps", f"{tm_total:,.0f}")

st.divider()

# ------------------------------------------------------------------
# 5  TABBED DASHBOARD VIEWS
# ------------------------------------------------------------------

overview_tab, sector_tab, ip_tab, growth_tab = st.tabs(
    ["Overview", "Sector Dive", "IP vs Financials", "Growth 2015‑18"]
)

# --------------------------------------------------
# 5.1  OVERVIEW TAB
# --------------------------------------------------

with overview_tab:
    st.subheader("Top 10 R&D Investors")
    top10 = df.nlargest(10, "rd")
    fig_bar = px.bar(
        top10,
        x="company_name",
        y="rd",
        text="rd",
        labels={"rd": "R&D (\u20AC Million)", "company_name": "Company"},
    )
    fig_bar.update_layout(xaxis_tickangle=-35, yaxis_title="R&D (\u20AC M)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("R&D Intensity vs Operating Margin")
    fig_scatter = px.scatter(
        df.dropna(subset=["rd_intensity", "op_margin", "ns"]),
        x="rd_intensity",
        y="op_margin",
        size="ns",
        color="nace2",
        hover_name="company_name",
        labels={
            "rd_intensity": "R&D Intensity (R&D / Net Sales)",
            "op_margin": "Operating Margin (OP / Net Sales)",
            "ns": "Net Sales (\u20AC M)",
            "nace2": "Sector",
        },
    )
    fig_scatter.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation heat‑map (numeric variables)
    st.subheader("Correlation Matrix of Key Metrics")
    corr_cols = [c for c in [
        "rd", "ns", "capex", "op", "emp", "rd_intensity", "op_margin", "ip5_total", "tm_total"
    ] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# 5.2  SECTOR DIVE TAB
# --------------------------------------------------

with sector_tab:
    st.subheader("Median R&D Intensity by Sector (Top 20)")
    sector_int = (
        df.groupby("nace2")[["rd_intensity"]]
        .median()
        .dropna()
        .sort_values("rd_intensity", ascending=False)
        .head(20)
        .reset_index()
    )
    fig_int = px.bar(
        sector_int,
        x="rd_intensity",
        y="nace2",
        orientation="h",
        labels={"rd_intensity": "Median R&D Intensity", "nace2": "Sector"},
    )
    fig_int.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat=".1%")
    st.plotly_chart(fig_int, use_container_width=True)

    st.subheader("Median Operating Margin by Sector (Top 20)")
    sector_op = (
        df.groupby("nace2")[["op_margin"]]
        .median()
        .dropna()
        .sort_values("op_margin", ascending=False)
        .head(20)
        .reset_index()
    )
    fig_op = px.bar(
        sector_op,
        x="op_margin",
        y="nace2",
        orientation="h",
        labels={"op_margin": "Median OP Margin", "nace2": "Sector"},
    )
    fig_op.update_layout(yaxis=dict(autorange="reversed"), xaxis_tickformat=".1%")
    st.plotly_chart(fig_op, use_container_width=True)

# --------------------------------------------------
# 5.3  IP VS FINANCIALS TAB
# --------------------------------------------------

with ip_tab:
    if "ip5_total" in df:
        st.subheader("R&D Spend vs IP5 Patent Filings")
        fig_ip = px.scatter(
            df,
            x="rd",
            y="ip5_total",
            size="ns",
            color="nace2",
            hover_name="company_name",
            labels={
                "rd": "R&D (\u20AC M)",
                "ip5_total": "IP5 Patent Families",
                "nace2": "Sector",
            },
        )
        st.plotly_chart(fig_ip, use_container_width=True)

    if "tm_total" in df:
        st.subheader("Net Sales vs Trademark Filings")
        fig_tm = px.scatter(
            df,
            x="ns",
            y="tm_total",
            size="rd",
            color="nace2",
            hover_name="company_name",
            labels={
                "ns": "Net Sales (\u20AC M)",
                "tm_total": "Trademark Apps",
                "nace2": "Sector",
            },
        )
        st.plotly_chart(fig_tm, use_container_width=True)

    if ("ip5_total" not in df) and ("tm_total" not in df):
        st.info("Patent or trademark columns not found in the dataset.")

# --------------------------------------------------
# 5.4  GROWTH 2015‑18 TAB
# --------------------------------------------------

with growth_tab:
    st.subheader("Top 10 Compound Annual Growth in R&D (2015‑18)")
    base_year = raw_df["year"].min()
    end_year = raw_df["year"].max()

    base = raw_df[raw_df["year"] == base_year][["company_id", "rd"]].set_index("company_id")
    end = raw_df[raw_df["year"] == end_year][["company_id", "rd"]].set_index("company_id")
    growth = end.join(base, lsuffix="_end", rsuffix="_base", how="inner")
    growth = growth[(growth["rd_base"] > 0) & (growth["rd_end"] > 0)]
    growth["rd_cagr"] = (growth["rd_end"] / growth["rd_base"]) ** (1 / (end_year - base_year)) - 1
    growth = growth.replace([np.inf, -np.inf], np.nan).dropna(subset=["rd_cagr"])

    top_growth = (
        growth.nlargest(10, "rd_cagr")
        .reset_index()
        .merge(raw_df[["company_id", "company_name"]].drop_duplicates(), on="company_id", how="left")
    )

    fig_growth = px.bar(
        top_growth,
        x="company_name",
        y="rd_cagr",
        text=top_growth["rd_cagr"].apply(lambda x: f"{x:.1%}"),
        labels={"rd_cagr": "CAGR R&D", "company_name": "Company"},
    )
    fig_growth.update_layout(xaxis_tickangle=-35, yaxis_tickformat=".0%", yaxis_title="CAGR (2015‑18)")
    st.plotly_chart(fig_growth, use_container_width=True)

    with st.expander("CAGR calculation details"):
        st.write(
            """The CAGR is computed for firms with positive R&D in both the base (2015) and end (2018)
            years. CAGR = (RD₍2018₎ / RD₍2015₎)^(1/3) − 1."""
        )

# ------------------------------------------------------------------
# 6  DATA TABLE & DOWNLOAD (GLOBAL FILTERED SAMPLE)
# ------------------------------------------------------------------

with st.expander("Show filtered data (current year)"):
    st.dataframe(df, use_container_width=True)

@st.cache_data
def to_csv_bytes(_df):
    return _df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download filtered CSV (current year)",
    to_csv_bytes(df),
    file_name="filtered_rd_dataset.csv",
    mime="text/csv",
)

# ------------------------------------------------------------------
# 7  FOOTER
# ------------------------------------------------------------------

st.caption("Data source: EC‑JRC / OECD COR&DIP © 2021 – *Top 2 000 Corporate R&D Investors*")
