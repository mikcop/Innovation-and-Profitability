# dashboard.py  ·  Streamlit 1.33+
# ------------------------------------------------------------
# Corporate R&D Innovation Dashboard – cleaned, cached, & documented
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

DATA_PATH = "panel_2015_2018.csv"

# ------------------------------------------------------------
# 1 · DATA LOAD & PRE‑PROCESS
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # fill NaN for count‑type columns, keep NaN for monetary zeros
    ip_cols = ["patCN", "patEP", "patJP", "patKR", "patUS", "TMnEU", "TMnUS"]
    df[ip_cols] = df[ip_cols].fillna(0)

    # core ratios
    df["rd_intensity"]   = df["rd"].div(df["ns"].replace({0: np.nan}))
    df["profit_margin"]  = df["op"].div(df["ns"].replace({0: np.nan}))
    df["patents_per_rd"] = df["patEP"].div(df["rd"].replace({0: np.nan}))
    df["revenue_per_employee"] = df["ns"].div(df["emp"].replace({0: np.nan}))

    return df

df = load_data(DATA_PATH)

# ISIC legend (top 10 blocks used in report)
ISIC4_MAP = {
    "10-12": "Food, Beverages & Tobacco",
    "20":    "Chemicals",
    "21":    "Pharmaceuticals",
    "26":    "Electronics & Optical",
    "27":    "Electrical Equipment",
    "28":    "Machinery & Equipment",
    "29-30": "Automotive & Transport",
    "58-60": "Publishing & Broadcasting",
    "61":    "Telecommunications",
    "62-63": "IT & Software Services",
}
df["isic4_desc"] = df["isic4"].astype(str).map(ISIC4_MAP).fillna(df["isic4"].astype(str))

# ------------------------------------------------------------
# 2 · SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.title("🔍 Filter Data")

if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

year       = st.sidebar.selectbox("Year", sorted(df["year"].unique()), index=3)
countries  = st.sidebar.multiselect("Country", df["ctry_code"].unique(),
                                    default=list(df["ctry_code"].unique()))
sectors    = st.sidebar.multiselect("Sector (ISIC4)", df["isic4"].unique(),
                                    default=list(df["isic4"].unique()))
rd_range   = st.sidebar.slider("R&D Intensity", 0.0, 10.0, (0.0, 2.0))
prof_range = st.sidebar.slider("Profit Margin", -2.0, 2.0, (-1.0, 1.0))

mask = (
    (df["year"] == year) &
    (df["ctry_code"].isin(countries)) &
    (df["isic4"].isin(sectors)) &
    (df["rd_intensity"].between(*rd_range)) &
    (df["profit_margin"].between(*prof_range))
)
filtered = df.loc[mask]

# ------------------------------------------------------------
# 3 · KPI CARDS
# ------------------------------------------------------------
st.title("🚀 Corporate R&D Innovation Dashboard")

k1, k2, k3 = st.columns(3)
k1.metric("Avg R&D Intensity",  f"{filtered['rd_intensity'].mean():.2%}")
k2.metric("Avg Profit Margin",  f"{filtered['profit_margin'].mean():.2%}")
top_firm = filtered.loc[filtered["rd"].idxmax(), "company_name"] if not filtered.empty else "—"
k3.metric("Top R&D Spender", top_firm)

# variable legend
with st.expander("📘 Variable Legend"):
    st.markdown(
        """
        | Code | Meaning |
        |------|---------|
        | **rd** | R&D investment (€ M) |
        | **ns** | Net sales (€ M) |
        | **capex** | Capital expenditures (€ M) |
        | **op** | Operating profits (€ M) |
        | **emp** | Employees |
        | **rd_intensity** | rd / ns |
        | **profit_margin** | op / ns |
        | **patEP** | EPO patent filings |
        """
    )

# ------------------------------------------------------------
# 4 · TIME‑SERIES (whole panel)
# ------------------------------------------------------------
st.subheader("📈 R&D & Finance – 2015 → 2018")

ts = (df.groupby("year")[["rd","ns","capex","op"]]
        .sum()
        .reset_index()
        .assign(rd_growth=lambda x: x["rd"].pct_change()*100,
                sales_growth=lambda x: x["ns"].pct_change()*100,
                capex_growth=lambda x: x["capex"].pct_change()*100,
                profit_growth=lambda x: x["op"].pct_change()*100))

fig_line = px.line(ts, x="year", y=["rd","ns","capex","op"],
                   labels={"value":"€ M","variable":"Metric"},
                   markers=True, template="plotly_white")
st.plotly_chart(fig_line, use_container_width=True)
st.dataframe(ts[["year","rd_growth","sales_growth",
                 "capex_growth","profit_growth"]], hide_index=True)

# ------------------------------------------------------------
# 5 · SCATTER rd_intensity vs profit_margin
# ------------------------------------------------------------
st.subheader("📊 Profit vs R&D Intensity")

if len(filtered) >= 5:
    fig_scatter = px.scatter(filtered, x="rd_intensity", y="profit_margin",
                             color="ctry_code", hover_data=["company_name"],
                             template="plotly_white")
    # quadratic curve
    try:
        p = np.poly1d(np.polyfit(filtered["rd_intensity"], filtered["profit_margin"], 2))
        xs = np.linspace(filtered["rd_intensity"].min(), filtered["rd_intensity"].max(), 200)
        fig_scatter.add_scatter(x=xs, y=p(xs), mode="lines", name="Quadratic fit")
    except Exception:
        pass
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough points for scatter plot.")

# ------------------------------------------------------------
# 6 · SECTOR BAR
# ------------------------------------------------------------
st.subheader("🏭 R&D Intensity by Sector")
sector = (df[df["year"] == year]
          .groupby("isic4_desc")["rd_intensity"]
          .mean()
          .nlargest(20)
          .reset_index())

fig_sector = px.bar(sector, y="isic4_desc", x="rd_intensity",
                    orientation="h", text_auto=".2f",
                    template="plotly_white", color="rd_intensity",
                    color_continuous_scale="Blues")
fig_sector.update_layout(xaxis_tickformat=".0%", yaxis_title="")
st.plotly_chart(fig_sector, use_container_width=True)

# ------------------------------------------------------------
# 7 · COUNTRY BAR (filtered)
# ------------------------------------------------------------
st.subheader("🌍 R&D Spend by Country")
country = (filtered.groupby("ctry_code")["rd"]
           .sum()
           .nlargest(20)
           .reset_index())
fig_country = px.bar(country, x="ctry_code", y="rd",
                     color="rd", labels={"rd":"R&D (€ M)"},
                     template="plotly_white")
st.plotly_chart(fig_country, use_container_width=True)

# ------------------------------------------------------------
# 8 · RANK SHIFTS
# ------------------------------------------------------------
st.subheader("📉 Rank Shifts 2015 → 2018")
r15 = df[df["year"] == 2015][["company_id","company_name","rd"]]
r15["rank_2015"] = r15["rd"].rank(ascending=False)
r18 = df[df["year"] == 2018][["company_id","rd"]]
r18["rank_2018"] = r18["rd"].rank(ascending=False)
shift = (r15.merge(r18, on="company_id")
             .assign(rank_shift=lambda x: x["rank_2015"] - x["rank_2018"]))

st.write("#### 🚀 Biggest Climbers")
st.dataframe(shift.nlargest(10, "rank_shift")
             [["company_name","rank_2015","rank_2018","rank_shift"]],
             hide_index=True)

st.write("#### 📉 Biggest Decliners")
st.dataframe(shift.nsmallest(10, "rank_shift")
             [["company_name","rank_2015","rank_2018","rank_shift"]],
             hide_index=True)

# ------------------------------------------------------------
# 9 · TOP‑10 FIRMS (filtered)
# ------------------------------------------------------------
st.subheader("🏢 Top 10 R&D Firms")
top_rd = filtered.nlargest(10, "rd")
fig_top = px.bar(top_rd, x="company_name", y="rd",
                 color="profit_margin", labels={"rd":"R&D (€ M)"},
                 template="plotly_white", color_continuous_scale="Bluered_r")
fig_top.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig_top, use_container_width=True)

# ------------------------------------------------------------
# 10 · INNOVATION EFFICIENCY
# ------------------------------------------------------------
st.subheader("📚 Innovation Efficiency (top 10)")
eff = (filtered.assign(patents_per_rd=lambda x: x["patEP"].div(x["rd"].replace({0:np.nan})))
                .sort_values("patents_per_rd", ascending=False)
                .head(10))

if eff.empty:
    st.info("No companies match the current filters.")
else:
    st.dataframe(eff[["company_name","ctry_code","rd","patEP",
                      "patents_per_rd","revenue_per_employee"]],
                 hide_index=True, format="%.2f")

# ------------------------------------------------------------
# 11 · ISIC LEGEND EXPANDER
# ------------------------------------------------------------
with st.expander("🏷️ ISIC Sector Legend (Top 10)"):
    for k, v in ISIC4_MAP.items():
        st.markdown(f"**{k}**  – {v}")

# ------------------------------------------------------------
# 11 · PATENTS ↔ SALES RELATIONSHIP
# ------------------------------------------------------------
st.subheader("🔗 Do more patents correlate with higher sales?")

if filtered.empty:
    st.info("No data after filters.")
else:
    # build a total‑patent count (better: switch to IP‑family count once available)
    filt = filtered.copy()
    filt["total_patents"] = (
        filt[["patCN","patEP","patJP","patKR","patUS"]].sum(axis=1)
    )

    # drop rows with zero or NaN patents / sales to avoid log issues
    sample = filt[(filt["total_patents"] > 0) & (filt["ns"] > 0)]

    if len(sample) < 5:
        st.info("Not enough observations with both patents and sales.")
    else:
        # Pearson correlation
        r = sample["total_patents"].corr(sample["ns"])
        st.metric("Pearson r (patents vs net sales)", f"{r:.2f}")

        # log‑log scatter with OLS trend‑line
        fig_corr = px.scatter(sample,
                              x="total_patents", y="ns",
                              hover_data=["company_name","ctry_code"],
                              trendline="ols", trendline_color_override="red",
                              template="plotly_white",
                              labels={"total_patents":"Total patents (CN+EP+JP+KR+US)",
                                      "ns":"Net sales (€ M)"})
        fig_corr.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig_corr, use_container_width=True)

        # optional: show stats of the fitted model
        results = px.get_trendline_results(fig_corr)
        ols_summary = results.iloc[0]["px_fit_results"].summary()
        with st.expander("Show OLS details"):
            st.text(ols_summary)
