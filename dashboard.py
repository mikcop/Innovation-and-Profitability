# dashboard.py  ·  Streamlit 1.33+
# ------------------------------------------------------------
# Corporate R&D Innovation Dashboard – with REGION labels
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

    # fill zero‑safe NaNs for count columns
    ip_cols = ["patCN","patEP","patJP","patKR","patUS","TMnEU","TMnUS"]
    df[ip_cols] = df[ip_cols].fillna(0)

    # ratios
    df["rd_intensity"]   = df["rd"].div(df["ns"].replace({0: np.nan}))
    df["profit_margin"]  = df["op"].div(df["ns"].replace({0: np.nan}))
    df["patents_per_rd"] = df["patEP"].div(df["rd"].replace({0: np.nan}))
    df["revenue_per_employee"] = df["ns"].div(df["emp"].replace({0: np.nan}))

    # ---  region labels  ---------------------------------------------------
    EU = {"AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR",
          "HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK",
          "SI","ES","SE"}
    ASIA = {"JP","KR","IN","ID","SG","MY","TH","VN","PH","TW","HK"}

    def region(code):
        if code == "US":
            return "USA"
        if code == "CN":
            return "China"
        if code in EU:
            return "EU"
        if code in ASIA:
            return "Asia"
        return "Other"

    df["region"] = df["ctry_code"].apply(region).astype("category")

    # ISIC legend (top 10)
    ISIC4_MAP = {
        "10-12":"Food, Beverages & Tobacco","20":"Chemicals","21":"Pharmaceuticals",
        "26":"Electronics & Optical","27":"Electrical Equipment","28":"Machinery & Equipment",
        "29-30":"Automotive & Transport","58-60":"Publishing & Broadcasting",
        "61":"Telecommunications","62-63":"IT & Software Services"
    }
    df["isic4_desc"] = df["isic4"].astype(str).map(ISIC4_MAP).fillna(df["isic4"].astype(str))
    return df

df = load_data(DATA_PATH)

# ------------------------------------------------------------
# 2 · SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.title("🔍 Filter Data")

if st.sidebar.button("Reset filters"):
    st.experimental_rerun()

year       = st.sidebar.selectbox("Year", sorted(df["year"].unique()), index=3)
regions    = st.sidebar.multiselect("Region", ["USA","China","EU","Asia","Other"],
                                    default=["USA","China","EU","Asia","Other"])
countries  = st.sidebar.multiselect("Country", df["ctry_code"].unique(),
                                    default=list(df["ctry_code"].unique()))
sectors    = st.sidebar.multiselect("Sector (ISIC4)", df["isic4"].unique(),
                                    default=list(df["isic4"].unique()))
rd_range   = st.sidebar.slider("R&D Intensity", 0.0, 10.0, (0.0, 2.0))
prof_range = st.sidebar.slider("Profit Margin", -2.0, 2.0, (-1.0, 1.0))

mask = (
    (df["year"] == year) &
    (df["region"].isin(regions)) &
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

k1,k2,k3 = st.columns(3)
k1.metric("Avg R&D Intensity",  f"{filtered['rd_intensity'].mean():.2%}")
k2.metric("Avg Profit Margin",  f"{filtered['profit_margin'].mean():.2%}")
top_firm = filtered.loc[filtered["rd"].idxmax(), "company_name"] if not filtered.empty else "—"
k3.metric("Top R&D Spender", top_firm)

# ------------------------------------------------------------
# 4 · TIME‑SERIES (whole panel)
# ------------------------------------------------------------
st.subheader("📈 R&D & Finance – 2015‑18")
ts = (df.groupby("year")[["rd","ns","capex","op"]].sum().reset_index())
fig_line = px.line(ts, x="year", y=["rd","ns","capex","op"],
                   markers=True, template="plotly_white",
                   labels={"value":"€ M","variable":"Metric"})
st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------------------------
# 5 · PROFIT vs R&D INTENSITY SCATTER (colour by region)
# ------------------------------------------------------------
st.subheader("📊 Profit vs R&D Intensity (by Region)")

if len(filtered) >= 5:
    fig_scatter = px.scatter(filtered, x="rd_intensity", y="profit_margin",
                             color="region",
                             hover_data=["company_name","ctry_code"],
                             template="plotly_white")
    # quadratic curve
    try:
        p = np.poly1d(np.polyfit(filtered["rd_intensity"], filtered["profit_margin"], 2))
        xr = np.linspace(filtered["rd_intensity"].min(), filtered["rd_intensity"].max(), 200)
        fig_scatter.add_scatter(x=xr, y=p(xr), mode="lines", name="Quadratic fit")
    except Exception: pass
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Not enough points for scatter plot.")

# ------------------------------------------------------------
# 6 · SECTOR BAR
# ------------------------------------------------------------
st.subheader("🏭 R&D Intensity by Sector")
sector = (df[df["year"] == year]
          .groupby("isic4_desc")["rd_intensity"]
          .mean().nlargest(20).reset_index())
fig_sector = px.bar(sector, y="isic4_desc", x="rd_intensity",
                    orientation="h", template="plotly_white",
                    text_auto=".2f", color="rd_intensity",
                    color_continuous_scale="Blues")
fig_sector.update_layout(xaxis_tickformat=".0%", yaxis_title="")
st.plotly_chart(fig_sector, use_container_width=True)

# ------------------------------------------------------------
# 7 · COUNTRY BAR (filtered)
# ------------------------------------------------------------
st.subheader("🌍 R&D Spend by Country")
country = (filtered.groupby("ctry_code")["rd"].sum()
           .nlargest(20).reset_index())
fig_country = px.bar(country, x="ctry_code", y="rd", color="region",
                     template="plotly_white", labels={"rd":"R&D (€ M)"})
st.plotly_chart(fig_country, use_container_width=True)

# ------------------------------------------------------------
# 8 · TOP‑10 FIRMS (filtered)
# ------------------------------------------------------------
st.subheader("🏢 Top 10 R&D Firms")
top_rd = filtered.nlargest(10, "rd")
fig_top = px.bar(top_rd, x="company_name", y="rd",
                 color="region", template="plotly_white",
                 labels={"rd":"R&D (€ M)"})
fig_top.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig_top, use_container_width=True)

# ------------------------------------------------------------
# 9 · PATENTS ↔ SALES (colour by region)
# ------------------------------------------------------------
st.subheader("🔗 Patents vs Net Sales (log‑log)")

if filtered.empty:
    st.info("No data after filters.")
else:
    sample = filtered.copy()
    sample["total_patents"] = sample[["patCN","patEP","patJP","patKR","patUS"]].sum(axis=1)
    sample = sample[(sample["total_patents"] > 0) & (sample["ns"] > 0)]
    if len(sample) < 5:
        st.info("Need at least 5 observations with patents & sales.")
    else:
        r = sample["total_patents"].corr(sample["ns"])
        st.metric("Pearson r", f"{r:.2f}")

        fig_corr = px.scatter(sample, x="total_patents", y="ns",
                              color="region", hover_data=["company_name","ctry_code"],
                              trendline="ols", trendline_color_override="red",
                              template="plotly_white",
                              labels={"total_patents":"Total patents (office counts)",
                                      "ns":"Net sales (€ M)"})
        fig_corr.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------
# 10 · REGION LEGEND EXPANDER
# ------------------------------------------------------------
with st.expander("🌐 Region Labels"):
    st.markdown("""
    * **USA** – United States  
    * **China** – Mainland China  
    * **EU** – 27 EU Member States  
    * **Asia** – Japan, Korea, India, ASEAN‑6, Taiwan, Hong Kong  
    * **Other** – all remaining economies
    """)

# ------------------------------------------------------------
# 11 · ISIC LEGEND EXPANDER
# ------------------------------------------------------------
with st.expander("🏷️ ISIC Sector Legend (Top 10)"):
    for k, v in {"10-12":"Food/Beverage/Tobacco","20":"Chemicals","21":"Pharma",
                 "26":"Electronics & Optical","27":"Electrical Equip.",
                 "28":"Machinery & Equip.","29-30":"Auto & Transport",
                 "58-60":"Publishing & Broadcast","61":"Telecom","62-63":"IT & Software"}.items():
        st.markdown(f"**{k}**  – {v}")

# ------------------------------------------------------------
# 12 · DOWNLOAD
# ------------------------------------------------------------
st.download_button("📥 Download filtered data",
                   data=filtered.to_csv(index=False).encode("utf‑8"),
                   file_name="filtered_panel.csv",
                   mime="text/csv")
