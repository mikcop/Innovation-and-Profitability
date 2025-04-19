"""
Corporate R&D Innovation Dashboard · rev 2025‑04
------------------------------------------------
Improvements
• one‑time cached load & tidy (incl. NaN→0 for IP counts)
• derived ratios pre‑computed once
• “Reset filters” button
• polynomial fit wrapped in a guard
• consistent Plotly template + rotated labels
• helper dictionaries for ISIC / country codes
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ----------------------------------------------------------------------
# 1 · DATA LOAD & PRE‑PROCESS
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # tidy types
    num_cols = ["rd", "ns", "capex", "op", "emp",
                "patCN", "patEP", "patJP", "patKR", "patUS", "TMnEU", "TMnUS"]
    df[num_cols] = df[num_cols].fillna(0)

    # core ratios
    df["rd_intensity"]   = df["rd"].div(df["ns"].replace({0: np.nan}))
    df["profit_margin"]  = df["op"].div(df["ns"].replace({0: np.nan}))
    df["patents_per_rd"] = df["patEP"].div(df["rd"].replace({0: np.nan}))
    df["revenue_per_employee"] = df["ns"].div(df["emp"].replace({0: np.nan}))

    # short ISIC legend (customise as needed)
    isic_map = {
        "21": "Pharma", "26": "Electronics & Optical", "29-30": "Auto & Transport",
        "62-63": "IT & Software", "10-12": "Food/Beverage/Tobacco", "20": "Chemicals",
        "27": "Electrical Equip.", "58-60": "Publishing/Broadcast", "28": "Machinery",
        "61": "Telecom"
    }
    df["isic4_desc"] = df["isic4"].astype(str).map(isic_map).fillna(df["isic4"].astype(str))
    return df

df = load_data("panel_2015_2018.csv")

# ----------------------------------------------------------------------
# 2 · SIDEBAR FILTERS
# ----------------------------------------------------------------------
st.sidebar.title("🔍 Filter Data")

if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

year       = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()), index=3)
countries  = st.sidebar.multiselect("Select Country", df["ctry_code"].unique(),
                                    default=list(df["ctry_code"].unique()))
sectors    = st.sidebar.multiselect("Select Sector (ISIC4)", df["isic4"].unique(),
                                    default=list(df["isic4"].unique()))
rd_range   = st.sidebar.slider("R&D Intensity Range", 0.0, 10.0, (0.0, 2.0))
prof_range = st.sidebar.slider("Profit Margin Range", -2.0, 2.0, (-1.0, 1.0))

mask = (
    (df["year"] == year) &
    (df["ctry_code"].isin(countries)) &
    (df["isic4"].isin(sectors)) &
    (df["rd_intensity"].between(*rd_range)) &
    (df["profit_margin"].between(*prof_range))
)
filtered = df.loc[mask]

# ----------------------------------------------------------------------
# 3 · KPI CARDS
# ----------------------------------------------------------------------
st.title("🚀 Corporate R&D Innovation Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Avg R&D Intensity",  f"{filtered['rd_intensity'].mean():.2%}")
col2.metric("Avg Profit Margin",  f"{filtered['profit_margin'].mean():.2%}")
top_firm = filtered.loc[filtered["rd"].idxmax(), "company_name"] if not filtered.empty else "—"
col3.metric("Top R&D Firm", top_firm)

# ----------------------------------------------------------------------
# 4 · TIME‑SERIES SECTION (entire panel, not filtered)
# ----------------------------------------------------------------------
st.subheader("📈 R&D and Financial Metrics (2015‑18)")

ts = (df.groupby("year")[["rd", "ns", "capex", "op"]]
        .sum()
        .reset_index()
        .assign(**{
            "rd_growth":     lambda x: x["rd"].pct_change()*100,
            "sales_growth":  lambda x: x["ns"].pct_change()*100,
            "capex_growth":  lambda x: x["capex"].pct_change()*100,
            "profit_growth": lambda x: x["op"].pct_change()*100,
        }))

fig_line = px.line(ts, x="year", y=["rd","ns","capex","op"],
                   markers=True, labels={"value":"€ million","variable":"Metric"},
                   template="plotly_white")
st.plotly_chart(fig_line, use_container_width=True)
st.dataframe(ts[["year","rd_growth","sales_growth","capex_growth","profit_growth"]],
             hide_index=True)

# ----------------------------------------------------------------------
# 5 · SCATTER (rd_intensity vs profit)
# ----------------------------------------------------------------------
st.subheader("📊 Profit vs R&D Intensity")

if len(filtered) >= 5:
    fig_scatter = px.scatter(filtered, x="rd_intensity", y="profit_margin",
                             hover_data=["company_name","ctry_code"],
                             color="ctry_code", template="plotly_white")
    # quadratic fit
    try:
        fit = np.poly1d(np.polyfit(filtered["rd_intensity"], filtered["profit_margin"], 2))
        xs = np.linspace(filtered["rd_intensity"].min(), filtered["rd_intensity"].max(), 200)
        fig_scatter.add_scatter(x=xs, y=fit(xs), mode="lines", name="Quadratic fit")
    except np.linalg.LinAlgError:
        pass
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Need at least 5 observations to display scatter & fit.")

# ----------------------------------------------------------------------
# 6 · SECTOR BAR (filtered year)
# ----------------------------------------------------------------------
st.subheader("🏭 Top R&D Intensity by Sector")

sector = (df[df["year"] == year]
          .groupby("isic4_desc")["rd_intensity"]
          .mean()
          .nlargest(20)
          .reset_index())

fig_sector = px.bar(sector, y="isic4_desc", x="rd_intensity",
                    orientation="h", text_auto=".2f",
                    labels={"rd_intensity":"R&D Intensity"},
                    template="plotly_white", color="rd_intensity",
                    color_continuous_scale="Blues")
fig_sector.update_layout(yaxis_title="", xaxis_tickformat=".0%")
st.plotly_chart(fig_sector, use_container_width=True)

# ----------------------------------------------------------------------
# 7 · COUNTRY R&D SPEND (filtered)
# ----------------------------------------------------------------------
st.subheader("🌍 R&D Investment by Country")

country = (filtered.groupby("ctry_code")["rd"]
           .sum()
           .nlargest(20)
           .reset_index())

fig_country = px.bar(country, x="ctry_code", y="rd",
                     color="rd", labels={"rd":"R&D (€ M)"},
                     template="plotly_white")
st.plotly_chart(fig_country, use_container_width=True)

# ----------------------------------------------------------------------
# 8 · RANK SHIFTS
# ----------------------------------------------------------------------
st.subheader("📉 R&D Rank Movers (2015 → 2018)")

r15 = df[df["year"] == 2015][["company_id","company_name","rd"]].assign(
        rank_2015=lambda x: x["rd"].rank(ascending=False))
r18 = df[df["year"] == 2018][["company_id","rd"]].assign(
        rank_2018=lambda x: x["rd"].rank(ascending=False))

shift = (r15.merge(r18, on="company_id")
             .assign(rank_shift=lambda x: x["rank_2015"] - x["rank_2018"]))

st.write("#### 🚀 Biggest Climbers")
st.dataframe(shift.nlargest(10, "rank_shift")
             [["company_name","rank_2015","rank_2018","rank_shift"]],
             hide_index=True)

st.write("#### 📉 Biggest Decliners")
st.dataframe(shift.nsmallest(10, "rank_shift")
             [["company_name","rank_2015","rank_2018","rank_shift"]],
             hide_index=True)

# ----------------------------------------------------------------------
# 9 · TOP‑10 FIRMS
# ----------------------------------------------------------------------
st.subheader("🏢 Top 10 R&D Firms (filtered)")

top_rd = filtered.nlargest(10, "rd")
fig_top = px.bar(top_rd, x="company_name", y="rd",
                 color="profit_margin", labels={"rd":"R&D (€ M)"},
                 template="plotly_white", color_continuous_scale="Bluered_r")
fig_top.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig_top, use_container_width=True)

# ----------------------------------------------------------------------
# 10 · INNOVATION EFFICIENCY TABLE
# ----------------------------------------------------------------------
st.subheader("📚 Innovation Efficiency Metrics")

eff = (filtered.assign(patents_per_rd=lambda x: x["patEP"].div(x["rd"].replace({0:np.nan})))
                .sort_values("patents_per_rd", ascending=False)
                .head(10))
st.dataframe(eff[["company_name","ctry_code","rd","patEP",
                  "patents_per_rd","revenue_per_employee"]],
             hide_index=True, format="%.2f")

# ----------------------------------------------------------------------
# 11 · DOWNLOAD
# ----------------------------------------------------------------------
st.download_button("📥 Download filtered data",
                   data=filtered.to_csv(index=False).encode("utf‑8"),
                   file_name="filtered_panel.csv",
                   mime="text/csv")
