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
def load_data(filepath="panel_2015_2018.csv"): # Ensure using CSV
    """Loads and preprocesses the R&D panel data."""
    try:
        df = pd.read_csv(filepath) # Use read_csv
    except FileNotFoundError:
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the correct directory and named correctly.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()


    # Define expected numeric columns
    numeric_cols_expected = [
        "rd", "ns", "capex", "op", "emp",
        "patCN", "patEP", "patJP", "patKR", "patUS",
        "TMnEU", "TMnUS"
    ]
    # Filter to only columns present in the DataFrame
    numeric_cols_present = [c for c in numeric_cols_expected if c in df.columns]

    # Convert to numeric, coercing errors
    if numeric_cols_present:
        df[numeric_cols_present] = df[numeric_cols_present].apply(pd.to_numeric, errors="coerce")

    # Basic data validation (check if essential columns are mostly present)
    essential_cols = ["year", "ctry_code", "nace2", "company_name", "rd", "ns", "op"]
    missing_essentials = [col for col in essential_cols if col not in df.columns]
    if missing_essentials:
        st.warning(f"Warning: Essential columns missing from data: {', '.join(missing_essentials)}")
        # Optionally, return empty df or handle differently
        # return pd.DataFrame()

    # Ensure 'year' is integer if it exists
    if 'year' in df.columns:
        # Attempt conversion, handle potential errors (e.g., if 'year' has non-int values)
        try:
            # Drop rows where year cannot be converted to int
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df.dropna(subset=['year'], inplace=True)
            df['year'] = df['year'].astype(int)
        except Exception as e:
            st.warning(f"Could not convert 'year' column to integer cleanly: {e}. Some year data might be lost.")
            # Keep as float or original type if conversion fails broadly
            pass


    return df

# --- Load Data ---
# Make sure the file path is correct
DATA_FILE = "panel_2015_2018.csv"
raw_df = load_data(DATA_FILE)

# --- Sidebar Filters ---
st.sidebar.title("Filter Panel")

if raw_df.empty:
    st.error(f"Dashboard cannot be loaded. Data file '{DATA_FILE}' might be missing or empty.")
    st.stop() # Stop execution if data loading failed critically

# Get available years, countries, sectors from the raw data
# Add checks for column existence before calling .unique()
years = sorted(raw_df["year"].dropna().unique()) if "year" in raw_df.columns else []
countries = sorted(raw_df["ctry_code"].dropna().unique()) if "ctry_code" in raw_df.columns else []
sectors = sorted(raw_df["nace2"].dropna().unique()) if "nace2" in raw_df.columns else []

# Handle cases where filters can't be populated
if not years:
    st.sidebar.error("No 'year' data found.")
    st.stop()
if not countries:
    st.sidebar.warning("No 'ctry_code' data found for filtering.")
    # Allow proceeding without country filter if needed, or stop
if not sectors:
    st.sidebar.warning("No 'nace2' data found for filtering.")
     # Allow proceeding without sector filter if needed, or stop


# Sidebar widgets
sel_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1 if years else 0)
# Use default=[] if countries/sectors might be empty initially
sel_countries = st.sidebar.multiselect("Select Countries", countries, default=countries if countries else [])
sel_sectors = st.sidebar.multiselect("Select Sectors (NACE2)", sectors, default=sectors if sectors else [])

# --- Data Filtering ---
# Build mask step-by-step for robustness
mask = pd.Series(True, index=raw_df.index) # Start with all true
if sel_year and 'year' in raw_df.columns:
     mask &= (raw_df["year"] == sel_year)
if sel_countries and 'ctry_code' in raw_df.columns:
     mask &= raw_df["ctry_code"].isin(sel_countries)
if sel_sectors and 'nace2' in raw_df.columns:
     mask &= raw_df["nace2"].isin(sel_sectors)

df_filtered_current_year = raw_df.loc[mask].copy()

# Handle case where filters result in no data for the selected year
if df_filtered_current_year.empty:
    st.warning("No data available for the selected filters and year.")
    # Don't stop here, allow Growth/Report tabs to show
    # st.stop() # Optionally stop if essential

# --- Calculated Metrics ---
# Add company identifier if missing for hover info
if 'company_name' not in df_filtered_current_year.columns:
     # Create a placeholder if 'company_name' is missing
     df_filtered_current_year['company_name'] = 'Company ' + df_filtered_current_year.index.astype(str)


def calculate_metrics(df_in):
    """Calculates derived metrics like R&D Intensity, Op Margin, IP totals."""
    if df_in is None or df_in.empty:
        return pd.DataFrame() # Return empty if input is invalid

    df_calc = df_in.copy()
    with np.errstate(divide="ignore", invalid="ignore"): # Suppress division warnings
        # Check if necessary columns exist before calculation
        if "rd" in df_calc.columns and "ns" in df_calc.columns:
            # Ensure 'ns' is not zero before division
            df_calc["rd_intensity"] = np.where(df_calc["ns"] != 0, df_calc["rd"] / df_calc["ns"], np.nan)
            df_calc["rd_intensity"] = df_calc["rd_intensity"].replace([np.inf, -np.inf], np.nan)
        else:
            df_calc["rd_intensity"] = np.nan

        if "op" in df_calc.columns and "ns" in df_calc.columns:
            df_calc["op_margin"] = np.where(df_calc["ns"] != 0, df_calc["op"] / df_calc["ns"], np.nan)
            df_calc["op_margin"] = df_calc["op_margin"].replace([np.inf, -np.inf], np.nan)
        else:
            df_calc["op_margin"] = np.nan

    # Calculate IP5 Patent Total
    pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in df_calc.columns]
    if pat_cols:
        # Fill NaNs with 0 before summing for patents/TMs if appropriate
        df_calc["ip5_total"] = df_calc[pat_cols].fillna(0).sum(axis=1)
    else:
        df_calc["ip5_total"] = 0 # Use 0 if columns don't exist

    # Calculate Total Trademark Applications
    tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in df_calc.columns]
    if tm_cols:
        df_calc["tm_total"] = df_calc[tm_cols].fillna(0).sum(axis=1)
    else:
        df_calc["tm_total"] = 0

    return df_calc

df = calculate_metrics(df_filtered_current_year)

# --- Aggregate KPIs ---
# Use .sum(skipna=True) for robustness against missing values within columns
kpi_rd = df["rd"].sum(skipna=True) if "rd" in df.columns and not df.empty else 0
kpi_ns = df["ns"].sum(skipna=True) if "ns" in df.columns and not df.empty else 0
kpi_op = df["op"].sum(skipna=True) if "op" in df.columns and not df.empty else 0
kpi_emp = df["emp"].sum(skipna=True) if "emp" in df.columns and not df.empty else 0

# Calculate aggregate intensities/margins carefully
kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns and kpi_ns != 0 else np.nan
kpi_op_margin = kpi_op / kpi_ns if kpi_ns and kpi_ns != 0 else np.nan

# Use calculated totals if available and df is not empty
kpi_pat_ip5 = df["ip5_total"].sum() if "ip5_total" in df.columns and not df.empty else 0
kpi_tm_total = df["tm_total"].sum() if "tm_total" in df.columns and not df.empty else 0

# Determine number of KPI columns based on available data
# Show base 4 KPIs, add IP/TM if they have non-zero values
num_kpis = 4
if kpi_pat_ip5 > 0: num_kpis += 1
if kpi_tm_total > 0: num_kpis += 1


st.header(f"R&D Investment Analysis ({sel_year})")

# Use map(str, ...) to safely join potentially numeric lists
country_str = ', '.join(map(str, sel_countries)) if sel_countries else "All"
sector_str = ', '.join(map(str, sel_sectors)) if sel_sectors else "All"
st.markdown(f"Based on {len(df)} companies matching filters: **Countries:** {country_str}, **Sectors:** {sector_str}")


kpi_cols = st.columns(num_kpis)

# Format KPIs carefully, handle NaN/zero
kpi_cols[0].metric("Total R&D (€ M)", f"{kpi_rd:,.0f}" if pd.notna(kpi_rd) and kpi_rd != 0 else "N/A")
kpi_cols[1].metric("Total Net Sales (€ M)", f"{kpi_ns:,.0f}" if pd.notna(kpi_ns) and kpi_ns != 0 else "N/A")
kpi_cols[2].metric("Avg. R&D Intensity", f"{kpi_rd_intensity:.2%}" if pd.notna(kpi_rd_intensity) else "N/A", help="Total R&D / Total Net Sales for filtered companies")
kpi_cols[3].metric("Avg. Operating Margin", f"{kpi_op_margin:.2%}" if pd.notna(kpi_op_margin) else "N/A", help="Total Operating Profit / Total Net Sales for filtered companies")

# Display IP KPIs conditionally
kpi_idx = 4
if kpi_pat_ip5 > 0:
    kpi_cols[kpi_idx].metric("Total IP5 Patent Apps", f"{kpi_pat_ip5:,.0f}", help="Sum of patents filed at CN, EP, JP, KR, US offices for filtered companies")
    kpi_idx += 1
if kpi_tm_total > 0:
    kpi_cols[kpi_idx].metric("Total EU/US TM Apps", f"{kpi_tm_total:,.0f}", help="Sum of trademarks filed at EUIPO and USPTO for filtered companies")

st.divider()

# --- Tabs for Detailed Analysis ---
tab_overview, tab_sector, tab_ip, tab_growth, tab_report = st.tabs([
    "📊 Overview", "🏭 Sector Deep Dive", "💡 IP vs Financials", "📈 Growth Analysis (vs 2015)", " JRC/OECD Report Insights"
])

# --- Tab 1: Overview ---
with tab_overview:
    if df.empty:
        st.info("No data to display for the selected filters in the Overview tab.")
    else:
        st.subheader("Company Landscape Overview")
        col1, col2 = st.columns(2)

        with col1:
            # Top Companies by R&D Spend
            st.markdown("**Top 10 Companies by R&D Spend (€ M)**")
            if "rd" in df.columns and df["rd"].notna().any():
                top_rd = df.nlargest(10, 'rd')[['company_name', 'rd']].reset_index(drop=True)
                if not top_rd.empty:
                    fig_top_rd = px.bar(top_rd, x='rd', y='company_name', orientation='h',
                                        title="Top 10 by R&D", labels={'rd': 'R&D Spend (€ M)', 'company_name': 'Company'},
                                        text='rd', height=400)
                    fig_top_rd.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_top_rd.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_top_rd, use_container_width=True)
                else:
                    st.info("No R&D data available for top company chart.")
            else:
                st.info("R&D column not available.")


            # Distribution of R&D Intensity
            st.markdown("**Distribution of R&D Intensity**")
            if 'rd_intensity' in df.columns and df['rd_intensity'].notna().any():
                plot_data_intensity = df['rd_intensity'].dropna()
                plot_data_intensity = plot_data_intensity[(plot_data_intensity > 0) & (plot_data_intensity < 1)] # Focus on 0-100%
                if not plot_data_intensity.empty:
                    fig_intensity_hist = px.histogram(plot_data_intensity, x='rd_intensity', nbins=30,
                                                    title="R&D Intensity Distribution (0-100%)",
                                                    labels={'rd_intensity': 'R&D Intensity (R&D/Net Sales)'})
                    fig_intensity_hist.update_layout(yaxis_title="Number of Companies", xaxis_tickformat='.1%')
                    st.plotly_chart(fig_intensity_hist, use_container_width=True)
                else:
                    st.info("No valid R&D Intensity data between 0% and 100% to display.")
            else:
                st.info("R&D Intensity data not available for distribution plot.")


        with col2:
            # Top Companies by Net Sales
            st.markdown("**Top 10 Companies by Net Sales (€ M)**")
            if "ns" in df.columns and df["ns"].notna().any():
                top_ns = df.nlargest(10, 'ns')[['company_name', 'ns']].reset_index(drop=True)
                if not top_ns.empty:
                    fig_top_ns = px.bar(top_ns, x='ns', y='company_name', orientation='h',
                                        title="Top 10 by Net Sales", labels={'ns': 'Net Sales (€ M)', 'company_name': 'Company'},
                                        text='ns', height=400)
                    fig_top_ns.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_top_ns.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide')
                    st.plotly_chart(fig_top_ns, use_container_width=True)
                else:
                    st.info("No Net Sales data available for top company chart.")
            else:
                st.info("Net Sales column not available.")

            # Distribution of Operating Margin
            st.markdown("**Distribution of Operating Margin**")
            if 'op_margin' in df.columns and df['op_margin'].notna().any():
                plot_data_margin = df['op_margin'].dropna()
                plot_data_margin = plot_data_margin[(plot_data_margin > -0.5) & (plot_data_margin < 0.5)] # Filter reasonable range
                if not plot_data_margin.empty:
                    fig_margin_hist = px.histogram(plot_data_margin, x='op_margin', nbins=30,
                                                title="Operating Margin Distribution (-50% to +50%)",
                                                labels={'op_margin': 'Operating Margin (Op Profit/Net Sales)'})
                    fig_margin_hist.update_layout(yaxis_title="Number of Companies", xaxis_tickformat='.1%')
                    st.plotly_chart(fig_margin_hist, use_container_width=True)
                else:
                    st.info("No valid Operating Margin data between -50% and +50% to display.")
            else:
                st.info("Operating Margin data not available for distribution plot.")

        # Display sample data
        st.subheader("Sample Data for Selected Filters")
        st.dataframe(df.head(10)) # Show sample of the filtered df

# --- Tab 2: Sector Deep Dive ---
with tab_sector:
    if df.empty:
        st.info("No data to display for the selected filters in the Sector Deep Dive tab.")
    elif 'nace2' not in df.columns:
        st.warning("Sector analysis requires the 'nace2' column, which is missing.")
    else:
        st.subheader(f"Sector Analysis for {sel_year}")

        # Group by sector and calculate aggregates
        # Use dropna=False to keep sectors even if they have missing values in some aggregate columns
        # Ensure columns exist before trying to aggregate
        agg_dict = {'company_count': ('company_name', 'count')}
        if "rd" in df.columns: agg_dict['total_rd'] = ('rd', 'sum')
        if "ns" in df.columns: agg_dict['total_ns'] = ('ns', 'sum')
        if "rd_intensity" in df.columns: agg_dict['median_rd_intensity'] = ('rd_intensity', 'median')
        if "op_margin" in df.columns: agg_dict['median_op_margin'] = ('op_margin', 'median')
        if "ip5_total" in df.columns: agg_dict['total_ip5'] = ('ip5_total', 'sum')

        if len(agg_dict) > 1: # Need more than just count for analysis
            sector_grouped = df.groupby('nace2', dropna=False).agg(**agg_dict).reset_index()

            # Calculate weighted average R&D Intensity for sectors if possible
            if 'total_rd' in sector_grouped.columns and 'total_ns' in sector_grouped.columns:
                 sector_grouped['avg_rd_intensity'] = np.where(
                     sector_grouped['total_ns'] != 0,
                     sector_grouped['total_rd'] / sector_grouped['total_ns'],
                     np.nan
                 ).astype(float) # Ensure float type
                 sector_grouped['avg_rd_intensity'] = sector_grouped['avg_rd_intensity'].replace([np.inf, -np.inf], np.nan)
            else:
                 sector_grouped['avg_rd_intensity'] = np.nan


            if not sector_grouped.empty:
                col1, col2 = st.columns(2)
                plot_height = 500 # Consistent height

                with col1:
                    # R&D Spend by Sector
                    if 'total_rd' in sector_grouped.columns:
                        st.markdown("**Total R&D Spend by Sector (€ M)**")
                        sector_rd_plot = sector_grouped.dropna(subset=['total_rd']).nlargest(15, 'total_rd') # Show top 15
                        if not sector_rd_plot.empty:
                            fig_sec_rd = px.bar(sector_rd_plot, x='total_rd', y='nace2', orientation='h',
                                                title="Top Sectors by Total R&D Spend",
                                                labels={'total_rd': 'Total R&D (€ M)', 'nace2': 'Sector (NACE2)'},
                                                hover_data=['company_count'], height=plot_height)
                            fig_sec_rd.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_sec_rd, use_container_width=True)
                        else:
                            st.info("No sector R&D data to display.")
                    else:
                         st.info("R&D data not available for sector plot.")

                    # Median R&D Intensity by Sector
                    if 'median_rd_intensity' in sector_grouped.columns:
                        st.markdown("**Median R&D Intensity by Sector**")
                        sector_int_plot = sector_grouped.dropna(subset=['median_rd_intensity']).nlargest(15, 'median_rd_intensity')
                        if not sector_int_plot.empty:
                            fig_sec_int = px.bar(sector_int_plot, x='median_rd_intensity', y='nace2', orientation='h',
                                                title="Top Sectors by Median R&D Intensity",
                                                labels={'median_rd_intensity': 'Median R&D Intensity', 'nace2': 'Sector (NACE2)'},
                                                hover_data=['company_count'], height=plot_height)
                            fig_sec_int.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
                            st.plotly_chart(fig_sec_int, use_container_width=True)
                        else:
                            st.info("No sector R&D Intensity data to display.")
                    else:
                         st.info("R&D Intensity data not available for sector plot.")


                with col2:
                    # Net Sales by Sector
                    if 'total_ns' in sector_grouped.columns:
                        st.markdown("**Total Net Sales by Sector (€ M)**")
                        sector_ns_plot = sector_grouped.dropna(subset=['total_ns']).nlargest(15, 'total_ns') # Show top 15
                        if not sector_ns_plot.empty:
                            fig_sec_ns = px.bar(sector_ns_plot, x='total_ns', y='nace2', orientation='h',
                                                title="Top Sectors by Total Net Sales",
                                                labels={'total_ns': 'Total Net Sales (€ M)', 'nace2': 'Sector (NACE2)'},
                                                hover_data=['company_count'], height=plot_height)
                            fig_sec_ns.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig_sec_ns, use_container_width=True)
                        else:
                            st.info("No sector Net Sales data to display.")
                    else:
                        st.info("Net Sales data not available for sector plot.")

                    # Median Operating Margin by Sector
                    if 'median_op_margin' in sector_grouped.columns:
                        st.markdown("**Median Operating Margin by Sector**")
                        sector_mar_plot = sector_grouped.dropna(subset=['median_op_margin']).nlargest(15, 'median_op_margin')
                        if not sector_mar_plot.empty:
                            fig_sec_mar = px.bar(sector_mar_plot, x='median_op_margin', y='nace2', orientation='h',
                                                title="Top Sectors by Median Operating Margin",
                                                labels={'median_op_margin': 'Median Operating Margin', 'nace2': 'Sector (NACE2)'},
                                                hover_data=['company_count'], height=plot_height)
                            fig_sec_mar.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_tickformat='.1%')
                            st.plotly_chart(fig_sec_mar, use_container_width=True)
                        else:
                            st.info("No sector Operating Margin data to display.")
                    else:
                         st.info("Operating Margin data not available for sector plot.")

                st.subheader("Sector Data Table")
                # Format columns for better readability in the table
                format_dict = {}
                if 'total_rd' in sector_grouped: format_dict['total_rd'] = '{:,.0f}'
                if 'total_ns' in sector_grouped: format_dict['total_ns'] = '{:,.0f}'
                if 'median_rd_intensity' in sector_grouped: format_dict['median_rd_intensity'] = '{:.2%}'
                if 'avg_rd_intensity' in sector_grouped: format_dict['avg_rd_intensity'] = '{:.2%}'
                if 'median_op_margin' in sector_grouped: format_dict['median_op_margin'] = '{:.2%}'
                if 'total_ip5' in sector_grouped: format_dict['total_ip5'] = '{:,.0f}'

                st.dataframe(
                    sector_grouped.sort_values(by='total_rd' if 'total_rd' in sector_grouped else 'company_count', ascending=False)
                    .reset_index(drop=True)
                    .style.format(format_dict, na_rep='N/A')
                )
            else:
                st.info("No data available to perform sector analysis for the current selection.")
        else:
            st.info("Insufficient data columns available for sector aggregation.")


# --- Tab 3: IP vs Financials ---
with tab_ip:
    st.subheader(f"IP Activity vs. Financial Performance ({sel_year})")

    # Check if IP and necessary financial columns are available
    ip_col = 'ip5_total'
    fin_cols = ['rd_intensity', 'op_margin', 'ns', 'rd']
    required_cols = [ip_col] + fin_cols + ['company_name', 'nace2']

    if df.empty:
         st.info("No data to display for the selected filters in the IP vs Financials tab.")
    elif all(col in df.columns for col in required_cols) and df[ip_col].notna().any():
        # Prepare data for scatter plots - drop rows where key variables for plotting are missing
        # Also handle potential NaNs in the 'size' column ('ns') explicitly
        scatter_df = df.dropna(subset=[ip_col, 'rd_intensity', 'op_margin', 'ns']).copy()
        # Replace any remaining NaNs in 'ns' with 0 for size mapping, or filter them out
        scatter_df['ns_size'] = scatter_df['ns'].fillna(0)
        # Ensure size is non-negative
        scatter_df = scatter_df[scatter_df['ns_size'] >= 0]

        if not scatter_df.empty:
            col1, col2 = st.columns(2)
            hover_list = ['rd', 'ns', 'op_margin', 'rd_intensity', ip_col]
            hover_data_subset = [c for c in hover_list if c in scatter_df.columns]


            with col1:
                st.markdown("**R&D Intensity vs. IP5 Patent Applications**")
                # Use log scale for patents if distribution is highly skewed and max > 0
                max_ip = scatter_df[ip_col].max()
                median_ip = scatter_df[ip_col].median()
                use_log_x = median_ip > 0 and max_ip > 10 * median_ip if pd.notna(max_ip) and pd.notna(median_ip) else False

                fig_ip_int = px.scatter(scatter_df, x=ip_col, y='rd_intensity',
                                        size='ns_size', color='nace2',  # Use ns_size
                                        hover_name='company_name',
                                        hover_data=hover_data_subset,
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
                                        size='ns_size', color='nace2', # Use ns_size
                                        hover_name='company_name',
                                        hover_data=hover_data_subset,
                                        log_x=use_log_x, # Use same log scale condition
                                        title="Operating Margin vs. IP5 Patents (Size = Net Sales)",
                                        labels={ip_col: f"IP5 Patent Applications {'(log scale)' if use_log_x else ''}",
                                                'op_margin': 'Operating Margin',
                                                'nace2': 'Sector'})
                fig_ip_mar.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_ip_mar, use_container_width=True)

            st.info("Bubble size represents Net Sales (€ M). Hover over points for company details. Log scale may be used for patents if data is highly skewed.")

        else:
            st.info("Insufficient valid data for IP vs. Financial scatter plots (requires non-missing Patents, R&D Intensity, Op Margin, and non-negative Net Sales).")
    else:
        missing_required = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
        st.warning(f"Cannot generate IP vs. Financials analysis. Missing or empty required columns: {', '.join(missing_required)}")


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
        mask_start = pd.Series(True, index=raw_df.index) # Start with all true
        if start_year and 'year' in raw_df.columns:
             mask_start &= (raw_df["year"] == start_year)
        if sel_countries and 'ctry_code' in raw_df.columns:
             mask_start &= raw_df["ctry_code"].isin(sel_countries)
        if sel_sectors and 'nace2' in raw_df.columns:
             mask_start &= raw_df["nace2"].isin(sel_sectors)

        df_start_year_filtered = raw_df.loc[mask_start].copy()

        if df_start_year_filtered.empty:
            st.warning(f"No data available for the selected filters in the baseline year ({start_year}). Cannot calculate growth.")
        else:
            # Calculate metrics for the start year
            if 'company_name' not in df_start_year_filtered.columns:
                 df_start_year_filtered['company_name'] = 'Company ' + df_start_year_filtered.index.astype(str)
            df_start = calculate_metrics(df_start_year_filtered)

            # --- Perform Growth Calculation ---
            # We need a common identifier. Assuming 'company_name' uniquely identifies companies across years.
            id_col = 'company_name' # Or choose a better unique ID if available
            if id_col not in df.columns or id_col not in df_start.columns:
                 st.error(f"Cannot perform growth analysis: Company identifier column '{id_col}' not found in both years' data.")
            elif df.empty or df_start.empty:
                 st.warning("No data available for comparison in either the start or selected year after filtering.")
            else:
                # Select relevant columns for comparison
                cols_for_growth = ['rd', 'ns', 'op', 'emp', 'ip5_total', 'tm_total']
                growth_cols = [col for col in cols_for_growth if col in df.columns and col in df_start.columns] # Only compare available cols

                if not growth_cols:
                     st.warning("No common columns found between years to calculate growth.")
                else:
                    # Merge data for the two years based on company identifier
                    df_growth = pd.merge(
                        df_start[[id_col] + growth_cols],
                        df[[id_col] + growth_cols],
                        on=id_col,
                        suffixes=(f'_{start_year}', f'_{sel_year}'),
                        how='inner' # Only keep companies present in BOTH years
                    )

                    if df_growth.empty:
                        st.warning(f"No matching companies found between {start_year} and {sel_year} with the current filters.")
                    else:
                        # Calculate growth rates
                        for col in growth_cols:
                            col_start = f'{col}_{start_year}'
                            col_end = f'{col}_{sel_year}'
                            growth_col_name = f'{col}_growth_pct'

                            start_values = df_growth[col_start]
                            end_values = df_growth[col_end]
                            df_growth[growth_col_name] = np.nan # Initialize

                            # Valid growth: (end - start) / start where start > 0
                            valid_mask = (start_values > 0) & start_values.notna() & end_values.notna()
                            df_growth.loc[valid_mask, growth_col_name] = (end_values[valid_mask] - start_values[valid_mask]) / start_values[valid_mask]

                            # Infinite growth: start <= 0 and end > 0
                            inf_mask = (start_values <= 0) & start_values.notna() & (end_values > 0) & end_values.notna()
                            df_growth.loc[inf_mask, growth_col_name] = np.inf

                            # -100% growth: start > 0 and end <= 0
                            neg_mask = (start_values > 0) & start_values.notna() & (end_values <= 0) & end_values.notna()
                            df_growth.loc[neg_mask, growth_col_name] = -1.0

                        st.success(f"Growth calculated for {len(df_growth)} companies found in both {start_year} and {sel_year}.")

                        # --- Display Growth KPIs ---
                        st.subheader(f"Overall Median Growth ({start_year} to {sel_year})")
                        growth_kpi_cols = st.columns(len(growth_cols))
                        for i, col in enumerate(growth_cols):
                            median_growth = df_growth[f'{col}_growth_pct'].replace([np.inf, -np.inf], np.nan).median()
                            growth_kpi_cols[i].metric(
                                f"Median {col.upper()} Growth",
                                f"{median_growth:.1%}" if pd.notna(median_growth) else "N/A",
                                help=f"Median percentage change in {col.upper()} from {start_year} to {sel_year} for matched companies."
                            )

                        # --- Visualize Growth ---
                        if 'rd_growth_pct' in df_growth.columns and 'nace2' in df.columns:
                             # Merge back sector info for visualization
                             df_growth_viz = pd.merge(df_growth[[id_col, 'rd_growth_pct']], df[[id_col, 'nace2']], on=id_col, how='left')
                             viz_data_growth = df_growth_viz.replace([np.inf, -np.inf], np.nan).dropna(subset=['rd_growth_pct', 'nace2'])
                             # Filter plot range e.g. -100% to +200% growth
                             viz_data_growth = viz_data_growth[(viz_data_growth['rd_growth_pct'] >= -1) & (viz_data_growth['rd_growth_pct'] <= 2)]

                             if not viz_data_growth.empty:
                                 st.markdown("**Distribution of R&D Growth (%) by Sector (-100% to +200%)**")
                                 # Get top N sectors by count for clarity
                                 top_sectors_growth = viz_data_growth['nace2'].value_counts().nlargest(15).index
                                 viz_data_growth_top = viz_data_growth[viz_data_growth['nace2'].isin(top_sectors_growth)]

                                 fig_growth_box = px.box(viz_data_growth_top, x='nace2', y='rd_growth_pct',
                                                         title=f"R&D Growth ({start_year}-{sel_year}) Distribution by Top 15 Sectors",
                                                         labels={'nace2': 'Sector (NACE2)', 'rd_growth_pct': 'R&D Growth (%)'},
                                                         points=False) # 'outliers' or False
                                 fig_growth_box.update_layout(yaxis_tickformat='.0%', xaxis_title="Sector (Top 15 by Company Count)")
                                 st.plotly_chart(fig_growth_box, use_container_width=True)
                             else:
                                 st.info("Insufficient data within the typical range (-100% to +200%) to display R&D growth distribution by sector.")
                        else:
                             st.info("R&D growth or Sector information not available for visualization.")

                        st.subheader("Company Growth Data Sample")
                        st.dataframe(df_growth.head())


# --- Tab 5: External Report Summary ---
with tab_report:
    st.subheader("📘 Insights from JRC–OECD Report (2017 Edition)") # Updated edition year
    st.warning("ℹ️ The information below summarizes key findings from the external report *World Corporate Top R&D Investors: Industrial Property Strategies in the Digital Economy* (2017 edition) and is **not** dynamically calculated from the filtered 2015-2018 data in this dashboard.")

    st.markdown("""
    Key findings from the report (based on Top 2000 Global R&D Investors, typically using data up to ~2014):

    *   **Dominance in ICT IP:** These firms account for approximately **75% of global ICT patents** and **60% of ICT design rights**. (Fig 1.1)
    *   **ICT Share of Portfolio:** Within their own IP portfolios, ICT represents almost **half of their patenting** activity and more than **a quarter of trademarks and designs**. (Fig 1.2)
    *   **Global & Diversified Operations:** Headquarters concentrated (US, JP, DE, UK, CN/TW dominate), but affiliates operate in >100 countries across ~9 industries on average. ~21% of affiliates operate in ICT. (Sec 2)
    *   **IP Strategy Insights:**
        *   Over half utilise **full IP bundles** (patents, trademarks, designs). (Sec 4.4)
        *   **'Computer & electronics'** is the most IP-intensive industry overall. (Sec 3)
        *   **USPTO** is a primary destination, especially for ICT patents (>30% filed there). (Sec 5.2)
        *   **Design rights** are crucial for product differentiation. (Sec 4.3)
    *   **Regional Specialization:**
        *   **Asia (KR, CN, TW)** shows strong specialization in ICT patenting. (Sec 4.1, Table 4.1)
        *   **EU & US** firms show a more balanced IP portfolio across sectors like health, environment, and energy. (Sec 4.1, Table 4.1)
    """)

    st.markdown("---")
    st.markdown("#### Illustrative Metrics from Report")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Global ICT IP Share (Fig 1.1)**")
        # Indicator for ICT Patent Share by Top 2000
        fig_indicator_pat = go.Figure(go.Indicator(
                mode="number", value=75, number={"suffix": "%"},
                title={"text": "Approx. Share of Global ICT Patents<br>by Top 2,000 R&D Firms", 'font_size': 14},
                domain={'y': [0.55, 1]}))
        fig_indicator_pat.add_trace(go.Indicator(
                mode="number", value=60, number={"suffix": "%"},
                title={"text": "Approx. Share of Global ICT Designs<br>by Top 2,000 R&D Firms", 'font_size': 14},
                 domain={'y': [0, 0.45]}))
        fig_indicator_pat.update_layout(height=250, margin=dict(t=50, b=10, l=0, r=0))
        st.plotly_chart(fig_indicator_pat, use_container_width=True)

    with col2:
        st.markdown("**ICT Share within Top Firms' Portfolio (Fig 1.2)**")
        # Pie Chart for internal portfolio focus
        pie_data_portfolio = pd.DataFrame({
            "IP Type": ["Patents related to ICT", "Designs related to ICT", "Trademarks related to ICT"],
            "Approx Portfolio Share": [45, 25, 25] # Approximated from Fig 1.2 text/visuals
        })
        pie_chart_portfolio = px.pie(pie_data_portfolio, names="IP Type", values="Approx Portfolio Share",
                           title="Approx. Share of ICT within<br>Top Firms' Own IP Portfolio",
                           height=250)
        pie_chart_portfolio.update_traces(textinfo='percent+label', insidetextorientation='radial')
        pie_chart_portfolio.update_layout(margin=dict(t=60, b=10, l=0, r=0), legend_title_text='IP Type') # legend_orientation="h"
        st.plotly_chart(pie_chart_portfolio, use_container_width=True)

    st.info("Reference: Daiko T., Dernis H., Dosso M., Gkotsis P., Squicciarini M., Vezzani A. (2017). *World Corporate Top R&D Investors: Industrial Property Strategies in the Digital Economy*. A JRC and OECD common report. Luxembourg: Publications Office of the European Union. EUR 28656 EN.")
