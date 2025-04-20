import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Corporate R&D Investors Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
px.defaults.template = "plotly_white"

# --- Data Loading ---
@st.cache_data(show_spinner="Loading data...")
def load_data(filepath="panel_2015_2018.csv"):
    """Loads and preprocesses the R&D panel data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the correct directory.")
        return pd.DataFrame() # Return empty DataFrame

    # Define expected numeric columns
    numeric_cols_expected = [
        "rd", "ns", "capex", "op", "emp",
        "patCN", "patEP", "patJP", "patKR", "patUS",
        "TMnEU", "TMnUS"
    ]
    # Filter to only columns present in the DataFrame
    numeric_cols_present = [c for c in numeric_cols_expected if c in df.columns]

    # Convert to numeric, coercing errors
    df[numeric_cols_present] = df[numeric_cols_present].apply(pd.to_numeric, errors="coerce")

    # Basic data validation (check if essential columns are mostly present)
    essential_cols = ["year", "ctry_code", "nace2", "company_name", "rd", "ns", "op"]
    missing_essentials = [col for col in essential_cols if col not in df.columns]
    if missing_essentials:
        st.warning(f"Warning: Essential columns missing from data: {', '.join(missing_essentials)}")
        # Optionally, return empty df or handle differently
        # return pd.DataFrame()

    # Ensure 'year' is integer
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)

    return df

# --- Load Data ---
raw_df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("Filter Panel")

if raw_df.empty:
    st.error("Dashboard cannot be loaded. Please check data availability and format.")
    st.stop() # Stop execution if data loading failed critically

# Get available years, countries, sectors from the raw data
years = sorted(raw_df["year"].dropna().unique())
countries = sorted(raw_df["ctry_code"].dropna().unique())
sectors = sorted(raw_df["nace2"].dropna().unique())

# Sidebar widgets
sel_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1)
sel_countries = st.sidebar.multiselect("Select Countries", countries, default=countries)
sel_sectors = st.sidebar.multiselect("Select Sectors (NACE2)", sectors, default=sectors)

# --- Data Filtering ---
# Filter data based on selections
mask = (
    (raw_df["year"] == sel_year)
    & raw_df["ctry_code"].isin(sel_countries)
    & raw_df["nace2"].isin(sel_sectors)
)
df_filtered_current_year = raw_df.loc[mask].copy()

# Handle case where filters result in no data
if df_filtered_current_year.empty:
    st.warning("No data available for the selected filters and year.")
    st.stop()

# --- Calculated Metrics ---
# Add company identifier if missing for hover info
if 'company_name' not in df_filtered_current_year.columns:
    df_filtered_current_year['company_name'] = 'Unknown Company ' + df_filtered_current_year.index.astype(str)


def calculate_metrics(df):
    """Calculates derived metrics like R&D Intensity, Op Margin, IP totals."""
    df_calc = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"): # Suppress division warnings
        # Check if necessary columns exist before calculation
        if "rd" in df_calc.columns and "ns" in df_calc.columns:
            df_calc["rd_intensity"] = (df_calc["rd"] / df_calc["ns"]).replace([np.inf, -np.inf], np.nan)
        else:
            df_calc["rd_intensity"] = np.nan

        if "op" in df_calc.columns and "ns" in df_calc.columns:
             df_calc["op_margin"] = (df_calc["op"] / df_calc["ns"]).replace([np.inf, -np.inf], np.nan)
        else:
            df_calc["op_margin"] = np.nan

    # Calculate IP5 Patent Total
    pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in df_calc.columns]
    if pat_cols:
        df_calc["ip5_total"] = df_calc[pat_cols].sum(axis=1, skipna=True) # Skipna=True handles missing patent data per company
    else:
        df_calc["ip5_total"] = np.nan # Set to NaN if no patent columns exist

    # Calculate Total Trademark Applications
    tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in df_calc.columns]
    if tm_cols:
        df_calc["tm_total"] = df_calc[tm_cols].sum(axis=1, skipna=True)
    else:
        df_calc["tm_total"] = np.nan

    return df_calc

df = calculate_metrics(df_filtered_current_year)

# --- Aggregate KPIs ---
# Use .sum(skipna=True) for robustness against missing values within columns
kpi_rd = df["rd"].sum(skipna=True) if "rd" in df.columns else 0
kpi_ns = df["ns"].sum(skipna=True) if "ns" in df.columns else 0
kpi_op = df["op"].sum(skipna=True) if "op" in df.columns else 0
kpi_emp = df["emp"].sum(skipna=True) if "emp" in df.columns else 0

# Calculate aggregate intensities/margins carefully
kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns else np.nan
kpi_op_margin = kpi_op / kpi_ns if kpi_ns else np.nan

kpi_pat_ip5 = df["ip5_total"].sum(skipna=True) if "ip5_total" in df.columns else np.nan
kpi_tm_total = df["tm_total"].sum(skipna=True) if "tm_total" in df.columns else np.nan

# Determine number of KPI columns based on available data
num_kpis = 4
if not pd.isna(kpi_pat_ip5): num_kpis += 1
if not pd.isna(kpi_tm_total): num_kpis += 1

# Corrected line 144
st.markdown(f"Based on {len(df)} companies matching filters: **Countries:** {', '.join(map(str, sel_countries))}, **Sectors:** {', '.join(map(str, sel_sectors))}")
kpi_cols = st.columns(num_kpis)

kpi_cols[0].metric("Total R&D (€ M)", f"{kpi_rd:,.0f}" if not pd.isna(kpi_rd) else "N/A")
kpi_cols[1].metric("Total Net Sales (€ M)", f"{kpi_ns:,.0f}" if not pd.isna(kpi_ns) else "N/A")
kpi_cols[2].metric("Avg. R&D Intensity", f"{kpi_rd_intensity:.2%}" if not pd.isna(kpi_rd_intensity) else "N/A", help="Total R&D / Total Net Sales")
kpi_cols[3].metric("Avg. Operating Margin", f"{kpi_op_margin:.2%}" if not pd.isna(kpi_op_margin) else "N/A", help="Total Operating Profit / Total Net Sales")

# Display IP KPIs conditionally
kpi_idx = 4
if not pd.isna(kpi_pat_ip5):
    kpi_cols[kpi_idx].metric("Total IP5 Patent Apps", f"{kpi_pat_ip5:,.0f}", help="Sum of patents filed at CN, EP, JP, KR, US offices")
    kpi_idx += 1
if not pd.isna(kpi_tm_total):
    kpi_cols[kpi_idx].metric("Total EU/US TM Apps", f"{kpi_tm_total:,.0f}", help="Sum of trademarks filed at EUIPO and USPTO")


st.divider()

# --- Tabs for Detailed Analysis ---
tab_overview, tab_sector, tab_ip, tab_growth, tab_report = st.tabs([
    "📊 Overview", "🏭 Sector Deep Dive", "💡 IP vs Financials", "📈 Growth Analysis (vs 2015)", " JRC/OECD Report Summary"
])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Company Landscape Overview")
    
    col1, col2 = st.columns(2)

    with col1:
        # Top Companies by R&D Spend
        st.markdown("**Top 10 Companies by R&D Spend (€ M)**")
        top_rd = df.nlargest(10, 'rd')[['company_name', 'rd']].reset_index(drop=True)
        if not top_rd.empty:
            fig_top_rd = px.bar(top_rd, x='rd', y='company_name', orientation='h',
                                title="Top 10 by R&D", labels={'rd': 'R&D Spend (€ M)', 'company_name': 'Company'},
                                height=400)
            fig_top_rd.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_rd, use_container_width=True)
        else:
            st.info("No R&D data available for top company chart.")

        # Distribution of R&D Intensity
        st.markdown("**Distribution of R&D Intensity**")
        if 'rd_intensity' in df.columns and df['rd_intensity'].notna().any():
            # Filter out extreme outliers for better visualization (e.g., > 100%)
            plot_data_intensity = df['rd_intensity'].dropna()
            plot_data_intensity = plot_data_intensity[(plot_data_intensity > 0) & (plot_data_intensity < 1)] # Focus on 0-100%
            if not plot_data_intensity.empty:
                fig_intensity_hist = px.histogram(plot_data_intensity, x='rd_intensity', nbins=30,
                                                  title="R&D Intensity Distribution (0-100%)",
                                                  labels={'rd_intensity': 'R&D Intensity (R&D/Net Sales)'})
                fig_intensity_hist.update_layout(yaxis_title="Number of Companies")
                st.plotly_chart(fig_intensity_hist, use_container_width=True)
            else:
                 st.info("No valid R&D Intensity data between 0% and 100% to display.")
        else:
            st.info("R&D Intensity data not available for distribution plot.")


    with col2:
        # Top Companies by Net Sales
        st.markdown("**Top 10 Companies by Net Sales (€ M)**")
        top_ns = df.nlargest(10, 'ns')[['company_name', 'ns']].reset_index(drop=True)
        if not top_ns.empty:
            fig_top_ns = px.bar(top_ns, x='ns', y='company_name', orientation='h',
                                title="Top 10 by Net Sales", labels={'ns': 'Net Sales (€ M)', 'company_name': 'Company'},
                                height=400)
            fig_top_ns.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_top_ns, use_container_width=True)
        else:
            st.info("No Net Sales data available for top company chart.")

        # Distribution of Operating Margin
        st.markdown("**Distribution of Operating Margin**")
        if 'op_margin' in df.columns and df['op_margin'].notna().any():
            # Filter out extreme outliers for better visualization
            plot_data_margin = df['op_margin'].dropna()
            # Filter reasonable range e.g., -50% to +50% margin
            plot_data_margin = plot_data_margin[(plot_data_margin > -0.5) & (plot_data_margin < 0.5)]
            if not plot_data_margin.empty:
                fig_margin_hist = px.histogram(plot_data_margin, x='op_margin', nbins=30,
                                               title="Operating Margin Distribution (-50% to +50%)",
                                               labels={'op_margin': 'Operating Margin (Op Profit/Net Sales)'})
                fig_margin_hist.update_layout(yaxis_title="Number of Companies")
                st.plotly_chart(fig_margin_hist, use_container_width=True)
            else:
                st.info("No valid Operating Margin data between -50% and +50% to display.")
        else:
            st.info("Operating Margin data not available for distribution plot.")

    # Display sample data
    st.subheader("Sample Data for Selected Filters")
    st.dataframe(df.head(10))

# --- Tab 2: Sector Deep Dive ---
with tab_sector:
    st.subheader(f"Sector Analysis for {sel_year}")

    # Group by sector and calculate aggregates
    # Use dropna=False to keep sectors even if they have missing values in some aggregate columns
    sector_grouped = df.groupby('nace2', dropna=False).agg(
        total_rd=('rd', 'sum'),
        total_ns=('ns', 'sum'),
        median_rd_intensity=('rd_intensity', 'median'),
        median_op_margin=('op_margin', 'median'),
        total_ip5=('ip5_total', 'sum'),
        company_count=('company_name', 'count')
    ).reset_index()

    # Calculate weighted average R&D Intensity for sectors
    sector_grouped['avg_rd_intensity'] = (sector_grouped['total_rd'] / sector_grouped['total_ns']).replace([np.inf, -np.inf], np.nan)

    if not sector_grouped.empty:
        col1, col2 = st.columns(2)

        with col1:
            # R&D Spend by Sector
            st.markdown("**Total R&D Spend by Sector (€ M)**")
            sector_rd_plot = sector_grouped.dropna(subset=['total_rd']).nlargest(15, 'total_rd') # Show top 15
            if not sector_rd_plot.empty:
                fig_sec_rd = px.bar(sector_rd_plot, x='total_rd', y='nace2', orientation='h',
                                    title="Top Sectors by Total R&D Spend",
                                    labels={'total_rd': 'Total R&D (€ M)', 'nace2': 'Sector (NACE2)'},
                                    hover_data={'company_count': True}, height=500)
                fig_sec_rd.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_sec_rd, use_container_width=True)
            else:
                st.info("No sector R&D data to display.")

            # Median R&D Intensity by Sector
            st.markdown("**Median R&D Intensity by Sector**")
            sector_int_plot = sector_grouped.dropna(subset=['median_rd_intensity']).nlargest(15, 'median_rd_intensity')
            if not sector_int_plot.empty:
                fig_sec_int = px.bar(sector_int_plot, x='median_rd_intensity', y='nace2', orientation='h',
                                     title="Top Sectors by Median R&D Intensity",
                                     labels={'median_rd_intensity': 'Median R&D Intensity', 'nace2': 'Sector (NACE2)'},
                                     hover_data={'company_count': True}, height=500)
                fig_sec_int.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
                st.plotly_chart(fig_sec_int, use_container_width=True)
            else:
                st.info("No sector R&D Intensity data to display.")


        with col2:
            # Net Sales by Sector
            st.markdown("**Total Net Sales by Sector (€ M)**")
            sector_ns_plot = sector_grouped.dropna(subset=['total_ns']).nlargest(15, 'total_ns') # Show top 15
            if not sector_ns_plot.empty:
                fig_sec_ns = px.bar(sector_ns_plot, x='total_ns', y='nace2', orientation='h',
                                    title="Top Sectors by Total Net Sales",
                                    labels={'total_ns': 'Total Net Sales (€ M)', 'nace2': 'Sector (NACE2)'},
                                    hover_data={'company_count': True}, height=500)
                fig_sec_ns.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_sec_ns, use_container_width=True)
            else:
                 st.info("No sector Net Sales data to display.")

            # Median Operating Margin by Sector
            st.markdown("**Median Operating Margin by Sector**")
            sector_mar_plot = sector_grouped.dropna(subset=['median_op_margin']).nlargest(15, 'median_op_margin')
            if not sector_mar_plot.empty:
                fig_sec_mar = px.bar(sector_mar_plot, x='median_op_margin', y='nace2', orientation='h',
                                     title="Top Sectors by Median Operating Margin",
                                     labels={'median_op_margin': 'Median Operating Margin', 'nace2': 'Sector (NACE2)'},
                                     hover_data={'company_count': True}, height=500)
                fig_sec_mar.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
                st.plotly_chart(fig_sec_mar, use_container_width=True)
            else:
                st.info("No sector Operating Margin data to display.")

        st.subheader("Sector Data Table")
        st.dataframe(sector_grouped.sort_values(by='total_rd', ascending=False).reset_index(drop=True))

    else:
        st.info("No data available to perform sector analysis for the current selection.")


# --- Tab 3: IP vs Financials ---
with tab_ip:
    st.subheader(f"IP Activity vs. Financial Performance ({sel_year})")

    # Check if IP and necessary financial columns are available
    ip_col = 'ip5_total'
    fin_cols = ['rd_intensity', 'op_margin', 'ns', 'rd']
    required_cols = [ip_col] + fin_cols + ['company_name', 'nace2']

    if all(col in df.columns for col in required_cols) and df[ip_col].notna().any():
        # Prepare data for scatter plots - drop rows where key variables for plotting are missing
        scatter_df = df.dropna(subset=[ip_col, 'rd_intensity', 'op_margin', 'ns']).copy()

        if not scatter_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**R&D Intensity vs. IP5 Patent Applications**")
                # Use log scale for patents if distribution is highly skewed
                use_log_x = scatter_df[ip_col].max() > 10 * scatter_df[ip_col].median() if scatter_df[ip_col].median() > 0 else False

                fig_ip_int = px.scatter(scatter_df, x=ip_col, y='rd_intensity',
                                        size='ns', color='nace2',  # Color by sector
                                        hover_name='company_name',
                                        hover_data=['rd', 'ns', 'op_margin'],
                                        log_x=use_log_x, # Use log scale conditionally
                                        title="R&D Intensity vs. IP5 Patents (Size = Net Sales)",
                                        labels={ip_col: f"IP5 Patent Applications {'(log scale)' if use_log_x else ''}",
                                                'rd_intensity': 'R&D Intensity',
                                                'nace2': 'Sector'})
                fig_ip_int.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_ip_int, use_container_width=True)

            with col2:
                st.markdown("**Operating Margin vs. IP5 Patent Applications**")
                fig_ip_mar = px.scatter(scatter_df, x=ip_col, y='op_margin',
                                        size='ns', color='nace2',
                                        hover_name='company_name',
                                        hover_data=['rd', 'ns', 'rd_intensity'],
                                        log_x=use_log_x, # Use same log scale condition
                                        title="Operating Margin vs. IP5 Patents (Size = Net Sales)",
                                        labels={ip_col: f"IP5 Patent Applications {'(log scale)' if use_log_x else ''}",
                                                'op_margin': 'Operating Margin',
                                                'nace2': 'Sector'})
                fig_ip_mar.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_ip_mar, use_container_width=True)

            st.info("Bubble size represents Net Sales (€ M). Hover over points for company details. Log scale may be used for patents if data is highly skewed.")

        else:
            st.info("Insufficient overlapping data for IP vs. Financial scatter plots (requires non-missing Patents, R&D Intensity, Op Margin, and Net Sales).")
    else:
        st.warning(f"Cannot generate IP vs. Financials analysis. Requires '{ip_col}' column and financial metrics (R&D Intensity, Op Margin, Net Sales) to be available in the data.")

# --- Tab 4: Growth Analysis ---
with tab_growth:
    st.subheader(f"Growth Analysis: {sel_year} vs. 2015")

    start_year = 2015 # Define the baseline year

    if start_year == sel_year:
        st.info(f"Selected year is the baseline year ({start_year}). Cannot calculate growth.")
    elif start_year not in years:
         st.warning(f"Baseline year {start_year} not found in the dataset. Cannot perform growth analysis.")
    else:
        # Filter data for the start year using the same country/sector filters
        mask_start = (
            (raw_df["year"] == start_year)
            & raw_df["ctry_code"].isin(sel_countries)
            & raw_df["nace2"].isin(sel_sectors)
        )
        df_start_year_filtered = raw_df.loc[mask_start].copy()

        if df_start_year_filtered.empty:
            st.warning(f"No data available for the selected filters in the baseline year ({start_year}). Cannot calculate growth.")
        else:
            # Calculate metrics for the start year
            df_start = calculate_metrics(df_start_year_filtered)

            # --- Perform Growth Calculation ---
            # We need a common identifier. Assuming 'company_name' uniquely identifies companies across years.
            # If not unique, another ID or combination (e.g., company_name + country) might be needed.
            id_col = 'company_name' # Or choose a better unique ID if available
            if id_col not in df.columns or id_col not in df_start.columns:
                 st.error(f"Cannot perform growth analysis: Company identifier column '{id_col}' not found in both years' data.")
            else:
                # Select relevant columns for comparison
                cols_for_growth = ['rd', 'ns', 'op', 'emp', 'ip5_total', 'tm_total']
                growth_cols = [col for col in cols_for_growth if col in df.columns and col in df_start.columns] # Only compare available cols

                # Merge data for the two years based on company identifier
                df_growth = pd.merge(
                    df_start[[id_col] + growth_cols],
                    df[[id_col] + growth_cols],
                    on=id_col,
                    suffixes=(f'_{start_year}', f'_{sel_year}')
                )

                if df_growth.empty:
                    st.warning("No matching companies found between the start year and selected year with the current filters.")
                else:
                    # Calculate growth rates
                    for col in growth_cols:
                        col_start = f'{col}_{start_year}'
                        col_end = f'{col}_{sel_year}'
                        growth_col_name = f'{col}_growth_pct'

                        # Calculate growth: (end - start) / start
                        # Handle division by zero or negative start values appropriately
                        start_values = df_growth[col_start]
                        end_values = df_growth[col_end]

                        # Initialize growth column with NaN
                        df_growth[growth_col_name] = np.nan

                        # Calculate growth only where start value is positive
                        valid_growth_mask = (start_values > 0) & start_values.notna() & end_values.notna()
                        df_growth.loc[valid_growth_mask, growth_col_name] = (end_values[valid_growth_mask] - start_values[valid_growth_mask]) / start_values[valid_growth_mask]

                        # Handle cases where start is 0 or negative and end is positive (infinite or large growth)
                        infinite_growth_mask = (start_values <= 0) & (end_values > 0) & start_values.notna() & end_values.notna()
                        df_growth.loc[infinite_growth_mask, growth_col_name] = np.inf # Or assign a large number like 999

                        # Handle cases where start is positive and end is zero or negative (set to -100% = -1.0)
                        neg_growth_mask = (start_values > 0) & (end_values <= 0) & start_values.notna() & end_values.notna()
                        df_growth.loc[neg_growth_mask, growth_col_name] = -1.0


                    st.success(f"Growth calculated for {len(df_growth)} companies found in both {start_year} and {sel_year}.")

                    # --- Display Growth KPIs ---
                    st.subheader(f"Overall Growth ({start_year} to {sel_year})")
                    growth_kpi_cols = st.columns(len(growth_cols))
                    for i, col in enumerate(growth_cols):
                         # Calculate median growth rate, ignoring inf/-inf
                         median_growth = df_growth[f'{col}_growth_pct'].replace([np.inf, -np.inf], np.nan).median()
                         growth_kpi_cols[i].metric(
                             f"Median {col.upper()} Growth",
                             f"{median_growth:.1%}" if not pd.isna(median_growth) else "N/A",
                             help=f"Median percentage change in {col.upper()} from {start_year} to {sel_year} for matched companies."
                         )

                    # --- Visualize Growth ---
                    # Example: Box plot of R&D Growth by Sector
                    if 'rd_growth_pct' in df_growth.columns:
                        # Need to merge back sector info
                        df_growth_viz = pd.merge(df_growth[[id_col, 'rd_growth_pct']], df[[id_col, 'nace2']], on=id_col, how='left')
                        # Filter out extreme outliers/infinite for visualization
                        viz_data_growth = df_growth_viz.replace([np.inf, -np.inf], np.nan).dropna(subset=['rd_growth_pct', 'nace2'])
                        # Focus plot range e.g. -100% to +200% growth
                        viz_data_growth = viz_data_growth[(viz_data_growth['rd_growth_pct'] >= -1) & (viz_data_growth['rd_growth_pct'] <= 2)]

                        if not viz_data_growth.empty:
                            st.markdown("**Distribution of R&D Growth (%) by Sector (-100% to +200%)**")
                            fig_growth_box = px.box(viz_data_growth, x='nace2', y='rd_growth_pct',
                                                    title=f"R&D Growth ({start_year}-{sel_year}) Distribution by Sector",
                                                    labels={'nace2': 'Sector (NACE2)', 'rd_growth_pct': 'R&D Growth (%)'},
                                                    points=False) # 'outliers' or False
                            fig_growth_box.update_layout(yaxis_tickformat='.0%')
                            st.plotly_chart(fig_growth_box, use_container_width=True)
                        else:
                            st.info("Insufficient data within the typical range (-100% to +200%) to display R&D growth distribution by sector.")
                    else:
                        st.info("R&D growth could not be calculated or visualized.")


                    st.subheader("Company Growth Data Sample")
                    st.dataframe(df_growth.head())


# --- Tab 5: External Report Summary ---
with tab_report:
    st.subheader("📘 Insights Summary from JRC–OECD Report (Static)")
    st.warning("ℹ️ The information below is based on the external *World Corporate Top R&D Investors* (2021) report and is **not** dynamically calculated from the filtered data in this dashboard.")

    st.markdown("""
    Key findings from the report focusing on the *Top 2,000 Global R&D Investors*:

    - **Dominance in ICT IP:** These firms account for approximately **75% of global ICT patents** and **60% of ICT design rights**.
    - **Global & Diversified Operations:** They operate in over **100 countries** on average and span roughly **9 different industrial sectors**.
    - **IP Strategy Insights:**
        - Over half utilize **full IP bundles** (patents, trademarks, designs).
        - Sectoral Differences: **Pharma** shows high R&D intensity per patent, while **Machinery & Electronics** lead in sheer volume of IP output.
        - Preferred Patent Offices: **USPTO** is primary for ICT patents, followed by **EPO** (Europe) and **SIPO/CNIPA** (China).
        - **Design rights** are highlighted as crucial for product differentiation, especially in ICT and Transport sectors.
    - **Regional Specialization:**
        - **Asia (notably KR, CN)** demonstrates strong specialization in ICT patenting.
        - **EU & US** firms show a more balanced IP portfolio across sectors like health, green technologies, and energy.
    """)

    st.markdown("---")
    st.markdown("#### Illustrative Metrics from Report")

    col1, col2 = st.columns(2)

    with col1:
        # Indicator for ICT Patent Share
        fig_indicator = go.Figure(
            go.Indicator(
                mode="number", # Removed delta as it's not a change over time here
                value=75,
                number={"suffix": "%"},
                title={"text": "Approx. Share of Global ICT Patents<br>by Top 2,000 R&D Firms (Source: Report)"},
                domain={'y': [0, 1], 'x': [0.25, 0.75]} # Centering
            )
        )
        fig_indicator.update_layout(height=200, margin=dict(t=50, b=40, l=0, r=0))
        st.plotly_chart(fig_indicator, use_container_width=True)

    with col2:
        # Pie Chart for Regional Focus (Illustrative)
        pie_data = pd.DataFrame({
            "Region Focus": ["Asia (Strong ICT Specialization)", "EU & US (More Diversified Portfolio)"],
            "Illustrative Share": [58, 42] # Example values reflecting report's findings
        })
        pie_chart = px.pie(pie_data, names="Region Focus", values="Illustrative Share",
                           title="Illustrative Regional Innovation Focus (Source: Report)",
                           height=250) # Adjusted height
        pie_chart.update_layout(margin=dict(t=50, b=0, l=0, r=0), legend_orientation="h")
        st.plotly_chart(pie_chart, use_container_width=True)


    st.info("Reference: EU JRC/OECD Report: *World Corporate Top R&D Investors: Industrial Property Strategies in the Digital Economy* (2021 edition). Consult the full report for detailed methodology and findings.")
