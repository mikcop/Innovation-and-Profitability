import warnings # <--- ADD THIS LINE
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.formula.api as smf 

# --- Page Config ---
st.set_page_config(
    page_title="Corporate R&D Investors Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Set Plotly template globally
px.defaults.template = "plotly_white"

# --- Data Loading ---
@st.cache_data(show_spinner="Loading data...")
def load_data(filepath="panel_2015_2018.csv"):
    """Loads and preprocesses the R&D panel data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: Data file '{filepath}' not found. Please ensure it's in the correct directory and named correctly.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()

    # Define expected numeric columns
    numeric_cols_expected = [
        "rd", "ns", "capex", "op", "emp",
        "patCN", "patEP", "patJP", "patKR", "patUS",
        "TMnEU", "TMnUS"
    ]
    numeric_cols_present = [c for c in numeric_cols_expected if c in df.columns]

    if numeric_cols_present:
        df[numeric_cols_present] = df[numeric_cols_present].apply(pd.to_numeric, errors="coerce")

    essential_cols = ["year", "ctry_code", "nace2", "company_name", "rd", "ns", "op"]
    missing_essentials = [col for col in essential_cols if col not in df.columns]
    if missing_essentials:
        st.warning(f"Warning: Essential columns missing: {', '.join(missing_essentials)}")

    if 'year' in df.columns:
        try:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df.dropna(subset=['year'], inplace=True)
            df['year'] = df['year'].astype(int)
        except Exception as e:
            st.warning(f"Could not convert 'year' column to integer cleanly: {e}.")

    return df

# --- Load Data ---
DATA_FILE = "panel_2015_2018.csv"
raw_df = load_data(DATA_FILE)

# --- Sidebar Filters ---
st.sidebar.title("Filter Panel")

if raw_df.empty:
    st.error(f"Dashboard cannot be loaded. Data file '{DATA_FILE}' might be missing or empty.")
    st.stop()

years = sorted(raw_df["year"].dropna().unique()) if "year" in raw_df.columns else []
countries = sorted(raw_df["ctry_code"].dropna().unique()) if "ctry_code" in raw_df.columns else []
sectors = sorted(raw_df["nace2"].dropna().unique()) if "nace2" in raw_df.columns else []

if not years:
    st.sidebar.error("No 'year' data found.")
    st.stop()
if not countries: st.sidebar.warning("No 'ctry_code' data found.")
if not sectors: st.sidebar.warning("No 'nace2' data found.")

sel_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1 if years else 0)
sel_countries = st.sidebar.multiselect("Select Countries", countries, default=countries if countries else [])
sel_sectors = st.sidebar.multiselect("Select Sectors (NACE2)", sectors, default=sectors if sectors else [])

# --- Data Filtering ---
mask = pd.Series(True, index=raw_df.index)
if sel_year and 'year' in raw_df.columns: mask &= (raw_df["year"] == sel_year)
if sel_countries and 'ctry_code' in raw_df.columns: mask &= raw_df["ctry_code"].isin(sel_countries)
if sel_sectors and 'nace2' in raw_df.columns: mask &= raw_df["nace2"].isin(sel_sectors)

df_filtered_current_year = raw_df.loc[mask].copy()

# --- Calculated Metrics ---
if 'company_name' not in df_filtered_current_year.columns:
     df_filtered_current_year['company_name'] = 'Company ' + df_filtered_current_year.index.astype(str)
     df = calculate_metrics(df_filtered_current_year)

def calculate_metrics(df_in):
    """Calculates derived metrics."""
    if df_in is None or df_in.empty: return pd.DataFrame()
    df_calc = df_in.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if "rd" in df_calc.columns and "ns" in df_calc.columns:
            df_calc["rd_intensity"] = np.where(df_calc["ns"] != 0, df_calc["rd"] / df_calc["ns"], np.nan)
            df_calc["rd_intensity"] = df_calc["rd_intensity"].replace([np.inf, -np.inf], np.nan)
        else: df_calc["rd_intensity"] = np.nan

        if "op" in df_calc.columns and "ns" in df_calc.columns:
            df_calc["op_margin"] = np.where(df_calc["ns"] != 0, df_calc["op"] / df_calc["ns"], np.nan)
            df_calc["op_margin"] = df_calc["op_margin"].replace([np.inf, -np.inf], np.nan)
        else: df_calc["op_margin"] = np.nan

    pat_cols = [c for c in ["patCN", "patEP", "patJP", "patKR", "patUS"] if c in df_calc.columns]
    df_calc["ip5_total"] = df_calc[pat_cols].fillna(0).sum(axis=1) if pat_cols else 0

    tm_cols = [c for c in ["TMnEU", "TMnUS"] if c in df_calc.columns]
    df_calc["tm_total"] = df_calc[tm_cols].fillna(0).sum(axis=1) if tm_cols else 0

    return df_calc

df = calculate_metrics(df_filtered_current_year)

# --- Aggregate KPIs ---
kpi_rd = df["rd"].sum(skipna=True) if "rd" in df.columns and not df.empty else 0
kpi_ns = df["ns"].sum(skipna=True) if "ns" in df.columns and not df.empty else 0
kpi_op = df["op"].sum(skipna=True) if "op" in df.columns and not df.empty else 0
kpi_emp = df["emp"].sum(skipna=True) if "emp" in df.columns and not df.empty else 0

kpi_rd_intensity = kpi_rd / kpi_ns if kpi_ns else np.nan
kpi_op_margin = kpi_op / kpi_ns if kpi_ns else np.nan

kpi_pat_ip5 = df["ip5_total"].sum() if "ip5_total" in df.columns and not df.empty else 0
kpi_tm_total = df["tm_total"].sum() if "tm_total" in df.columns and not df.empty else 0

num_kpis = 4
if kpi_pat_ip5 > 0: num_kpis += 1
if kpi_tm_total > 0: num_kpis += 1

st.header(f"R&D Investment Analysis ({sel_year})")
country_str = ', '.join(map(str, sel_countries)) if sel_countries else "All"
sector_str = ', '.join(map(str, sel_sectors)) if sel_sectors else "All"
st.markdown(f"Based on **{len(df)}** companies matching filters: **Countries:** {country_str}, **Sectors:** {sector_str}")

kpi_cols = st.columns(num_kpis)
kpi_cols[0].metric("Total R&D (€ M)", f"{kpi_rd:,.0f}" if pd.notna(kpi_rd) and kpi_rd != 0 else "N/A")
kpi_cols[1].metric("Total Net Sales (€ M)", f"{kpi_ns:,.0f}" if pd.notna(kpi_ns) and kpi_ns != 0 else "N/A")
kpi_cols[2].metric("Avg. R&D Intensity", f"{kpi_rd_intensity:.2%}" if pd.notna(kpi_rd_intensity) else "N/A", help="Total R&D / Total Net Sales for filtered companies")
kpi_cols[3].metric("Avg. Operating Margin", f"{kpi_op_margin:.2%}" if pd.notna(kpi_op_margin) else "N/A", help="Total Operating Profit / Total Net Sales for filtered companies")
kpi_idx = 4
if kpi_pat_ip5 > 0:
    kpi_cols[kpi_idx].metric("Total IP5 Patent Apps", f"{kpi_pat_ip5:,.0f}", help="Sum of patents filed at CN, EP, JP, KR, US offices for filtered companies")
    kpi_idx += 1
if kpi_tm_total > 0:
    kpi_cols[kpi_idx].metric("Total EU/US TM Apps", f"{kpi_tm_total:,.0f}", help="Sum of trademarks filed at EUIPO and USPTO for filtered companies")

st.divider()

# --- Tabs for Detailed Analysis ---
tab_overview, tab_sector, tab_ip, tab_growth, tab_framework, tab_regression, tab_report = st.tabs([
    "📊 Overview", "🏭 Sector Deep Dive", "💡 IP vs Financials", "📈 Growth Analysis",
    "📝 Analysis Framework", "📉 Regression Analysis", "📚 JRC/OECD Report (2017)" # Added Regression Tab
])


# --- Tab 1: Overview ---
with tab_overview:
    if df.empty:
        st.info("No data to display for the selected filters in the Overview tab.")
    else:
        st.subheader("Company Landscape Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top 10 Companies by R&D Spend (€ M)**")
            if "rd" in df.columns and df["rd"].notna().any():
                top_rd = df.nlargest(10, 'rd')[['company_name', 'rd']].reset_index(drop=True)
                if not top_rd.empty:
                    fig_top_rd = px.bar(top_rd, x='rd', y='company_name', orientation='h',
                                        title="Top 10 by R&D", labels={'rd': 'R&D Spend (€ M)', 'company_name': 'Company'},
                                        text='rd', height=400)
                    fig_top_rd.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_top_rd.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide',
                                            xaxis_title="R&D Spend (€ M)")
                    st.plotly_chart(fig_top_rd, use_container_width=True)
                else: st.info("No R&D data for top company chart.")
            else: st.info("R&D column not available.")

            st.markdown("**Distribution of R&D Intensity**")
            if 'rd_intensity' in df.columns and df['rd_intensity'].notna().any():
                plot_data_intensity = df['rd_intensity'].dropna()
                # Focus on a reasonable range, e.g., 0% to 100%
                plot_data_intensity_filtered = plot_data_intensity[(plot_data_intensity >= 0) & (plot_data_intensity <= 1)]
                if not plot_data_intensity_filtered.empty:
                    fig_intensity_hist = px.histogram(plot_data_intensity_filtered, x='rd_intensity', nbins=30,
                                                    title="R&D Intensity Distribution (0% to 100%)",
                                                    labels={'rd_intensity': 'R&D Intensity (R&D/Net Sales)'})
                    fig_intensity_hist.update_layout(yaxis_title="Number of Companies", xaxis_tickformat='.1%')
                    st.plotly_chart(fig_intensity_hist, use_container_width=True)
                    # Add note about filtering if some data was excluded
                    excluded_count = len(plot_data_intensity) - len(plot_data_intensity_filtered)
                    if excluded_count > 0:
                         st.caption(f"Note: {excluded_count} companies with R&D Intensity outside the 0-100% range are excluded from this chart for clarity.")
                else: st.info("No valid R&D Intensity data between 0% and 100% to display.")
            else: st.info("R&D Intensity data not available.")

        with col2:
            st.markdown("**Top 10 Companies by Net Sales (€ M)**")
            if "ns" in df.columns and df["ns"].notna().any():
                top_ns = df.nlargest(10, 'ns')[['company_name', 'ns']].reset_index(drop=True)
                if not top_ns.empty:
                    fig_top_ns = px.bar(top_ns, x='ns', y='company_name', orientation='h',
                                        title="Top 10 by Net Sales", labels={'ns': 'Net Sales (€ M)', 'company_name': 'Company'},
                                        text='ns', height=400)
                    fig_top_ns.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    fig_top_ns.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide',
                                            xaxis_title="Net Sales (€ M)")
                    st.plotly_chart(fig_top_ns, use_container_width=True)
                else: st.info("No Net Sales data for top company chart.")
            else: st.info("Net Sales column not available.")

            st.markdown("**Distribution of Operating Margin**")
            if 'op_margin' in df.columns and df['op_margin'].notna().any():
                plot_data_margin = df['op_margin'].dropna()
                 # Focus on a reasonable range, e.g., -50% to +50%
                plot_data_margin_filtered = plot_data_margin[(plot_data_margin >= -0.5) & (plot_data_margin <= 0.5)]
                if not plot_data_margin_filtered.empty:
                    fig_margin_hist = px.histogram(plot_data_margin_filtered, x='op_margin', nbins=30,
                                                title="Operating Margin Distribution (-50% to +50%)",
                                                labels={'op_margin': 'Operating Margin (Op Profit/Net Sales)'})
                    fig_margin_hist.update_layout(yaxis_title="Number of Companies", xaxis_tickformat='.1%')
                    st.plotly_chart(fig_margin_hist, use_container_width=True)
                    excluded_count_margin = len(plot_data_margin) - len(plot_data_margin_filtered)
                    if excluded_count_margin > 0:
                         st.caption(f"Note: {excluded_count_margin} companies with Operating Margin outside the -50% to +50% range are excluded from this chart for clarity.")
                else: st.info("No valid Operating Margin data between -50% and +50% to display.")
            else: st.info("Operating Margin data not available.")

        st.subheader("Sample Data for Selected Filters")
        st.dataframe(df.head(10).style.format("{:,.2f}", subset=pd.IndexSlice[:, ['rd_intensity', 'op_margin']], na_rep='N/A')
                                          .format("{:,.0f}", subset=pd.IndexSlice[:, ['rd', 'ns', 'op', 'emp', 'ip5_total', 'tm_total']], na_rep='N/A'))
# --- NEW: Top 10 Regression Plot ---
        st.subheader("Relationship for Top 10 R&D Investors")
        st.markdown("*(Based on companies with highest R&D Spend within current filters)*")

        # Define variables for the plot
        x_var = "rd_intensity"
        y_var = "op_margin"
        rank_col = "rd" # Rank by R&D Spend

        # Check if necessary columns exist
        plot_cols = [x_var, y_var, rank_col, 'company_name']
        if not all(col in df.columns for col in plot_cols):
             missing_plot_cols = [c for c in plot_cols if c not in df.columns]
             st.warning(f"Cannot create Top 10 regression plot. Missing columns: {', '.join(missing_plot_cols)}")
        else:
            # Get top 10 based on ranking column
            df_top10 = df.nlargest(10, rank_col).copy()

            # Prepare data for plotting (drop NaNs for x and y vars)
            df_plot = df_top10[[x_var, y_var, 'company_name']].dropna()

            # Need at least 2 points to plot a line
            if len(df_plot) < 2:
                st.info("Not enough valid data points among the Top 10 R&D investors to create a regression plot.")
            else:
                st.markdown(f"**{y_var.replace('_', ' ').title()} vs. {x_var.replace('_', ' ').title()}**")

                # Create scatter plot with OLS trendline
                fig_top10_reg = px.scatter(
                    df_plot,
                    x=x_var,
                    y=y_var,
                    title=f"{y_var.replace('_', ' ').title()} vs. {x_var.replace('_', ' ').title()} for Top 10 by R&D Spend",
                    trendline="ols",  # Add Ordinary Least Squares regression line
                    trendline_scope="overall", # Fit line to all points shown
                    trendline_color_override="red", # Make line stand out
                    labels={
                        x_var: x_var.replace('_', ' ').title(),
                        y_var: y_var.replace('_', ' ').title()
                    },
                    hover_name="company_name",
                    text="company_name" # Optionally label points directly
                )

                # Customize layout
                fig_top10_reg.update_traces(textposition='top center', textfont_size=10) # Adjust text label appearance
                fig_top10_reg.update_layout(
                     xaxis_tickformat='.1%', # Format x-axis if it's a percentage
                     yaxis_tickformat='.1%', # Format y-axis if it's a percentage
                     # Consider setting axis ranges if needed, similar to the IP vs Fin tab
                     # yaxis_range=[-0.5, 0.5], # Example range for op_margin
                     # xaxis_range=[0, 1],     # Example range for rd_intensity
                 )

                st.plotly_chart(fig_top10_reg, use_container_width=True)
                st.caption("Shows the relationship and linear trendline for the 10 companies with the highest R&D spending within the current filter selection.")

# --- Tab 2: Sector Deep Dive ---
with tab_sector:
    if df.empty:
        st.info("No data to display for the selected filters in the Sector Deep Dive tab.")
    elif 'nace2' not in df.columns:
        st.warning("Sector analysis requires the 'nace2' column, which is missing.")
    else:
        st.subheader(f"Sector Analysis for {sel_year}")

        # --- Aggregation (Assume sector_grouped is calculated as before) ---
        agg_dict = {'company_count': ('company_name', 'count')}
        if "rd" in df.columns: agg_dict['total_rd'] = ('rd', 'sum')
        if "ns" in df.columns: agg_dict['total_ns'] = ('ns', 'sum')
        if "rd_intensity" in df.columns: agg_dict['median_rd_intensity'] = ('rd_intensity', 'median')
        if "op_margin" in df.columns: agg_dict['median_op_margin'] = ('op_margin', 'median')
        if "ip5_total" in df.columns: agg_dict['total_ip5'] = ('ip5_total', 'sum')

        if len(agg_dict) > 1:
            with warnings.catch_warnings():
                 warnings.filterwarnings('ignore', r'Mean of empty slice', category=RuntimeWarning)
                 sector_grouped = df.groupby('nace2', dropna=False).agg(**agg_dict).reset_index()

            if 'total_rd' in sector_grouped.columns and 'total_ns' in sector_grouped.columns:
                 sector_grouped['avg_rd_intensity'] = np.where(
                     sector_grouped['total_ns'] != 0,
                     sector_grouped['total_rd'] / sector_grouped['total_ns'], np.nan
                 ).astype(float)
                 sector_grouped['avg_rd_intensity'] = sector_grouped['avg_rd_intensity'].replace([np.inf, -np.inf], np.nan)
            else: sector_grouped['avg_rd_intensity'] = np.nan

            # --- Plotting and Table ---
            if not sector_grouped.empty:
                col1, col2 = st.columns(2)
                plot_height = 500 # Adjust height as needed

                # --- Column 1 Plots ---
                with col1:
                    # R&D Spend by Sector (Keep as is - usually works well)
                    if 'total_rd' in sector_grouped.columns:
                        st.markdown("**Total R&D Spend by Sector (€ M)**")
                        sector_rd_plot = sector_grouped.dropna(subset=['total_rd']).nlargest(15, 'total_rd')
                        if not sector_rd_plot.empty:
                            # Convert nace2 to string for categorical plotting
                            sector_rd_plot['nace2_str'] = sector_rd_plot['nace2'].astype(str)
                            fig_sec_rd = px.bar(sector_rd_plot, x='total_rd', y='nace2_str', orientation='h',
                                                title="Top 15 Sectors by Total R&D Spend",
                                                labels={'total_rd': 'Total R&D (€ M)', 'nace2_str': 'Sector (NACE2)'},
                                                hover_data=['company_count', 'nace2'], height=plot_height, text='total_rd')
                            fig_sec_rd.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                            fig_sec_rd.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide')
                            st.plotly_chart(fig_sec_rd, use_container_width=True)
                        else: st.info("No valid sector R&D data to display.")
                    else: st.info("R&D data not available.")

                    # Median R&D Intensity by Sector (REVISED)
                    if 'median_rd_intensity' in sector_grouped.columns:
                        st.markdown("**Median R&D Intensity by Sector**")
                        sector_int_plot = sector_grouped.dropna(subset=['median_rd_intensity']).nlargest(15, 'median_rd_intensity')
                        if not sector_int_plot.empty:
                            # Convert nace2 to string for categorical plotting
                            sector_int_plot['nace2_str'] = sector_int_plot['nace2'].astype(str)
                            fig_sec_int = px.bar(sector_int_plot,
                                                 x='median_rd_intensity',
                                                 y='nace2_str', # Use string version for y-axis
                                                 orientation='h',
                                                 title="Top 15 Sectors by Median R&D Intensity",
                                                 labels={'median_rd_intensity': 'Median R&D Intensity', 'nace2_str': 'Sector (NACE2)'},
                                                 hover_data=['company_count', 'nace2'], # Show original nace2 on hover
                                                 height=plot_height,
                                                 text='median_rd_intensity' # Add text label
                                                 )
                            # Format the text label as percentage
                            fig_sec_int.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                            fig_sec_int.update_layout(
                                yaxis={'categoryorder':'total ascending', 'type': 'category'}, # Explicitly set category type
                                xaxis_tickformat='.1%', # Format x-axis as percentage
                                uniformtext_minsize=8, uniformtext_mode='hide' # Adjust text if bars overlap
                                )
                            st.plotly_chart(fig_sec_int, use_container_width=True)
                        else: st.info("No valid sector R&D Intensity data to display.")
                    else: st.info("R&D Intensity data not available.")

                # --- Column 2 Plots ---
                with col2:
                    # Net Sales by Sector (Keep as is)
                    if 'total_ns' in sector_grouped.columns:
                         st.markdown("**Total Net Sales by Sector (€ M)**")
                         sector_ns_plot = sector_grouped.dropna(subset=['total_ns']).nlargest(15, 'total_ns')
                         if not sector_ns_plot.empty:
                             # Convert nace2 to string for categorical plotting
                             sector_ns_plot['nace2_str'] = sector_ns_plot['nace2'].astype(str)
                             fig_sec_ns = px.bar(sector_ns_plot, x='total_ns', y='nace2_str', orientation='h',
                                                 title="Top 15 Sectors by Total Net Sales",
                                                 labels={'total_ns': 'Total Net Sales (€ M)', 'nace2_str': 'Sector (NACE2)'},
                                                 hover_data=['company_count', 'nace2'], height=plot_height, text='total_ns')
                             fig_sec_ns.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                             fig_sec_ns.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide')
                             st.plotly_chart(fig_sec_ns, use_container_width=True)
                         else: st.info("No valid sector Net Sales data to display.")
                    else: st.info("Net Sales data not available.")

                    # Median Operating Margin by Sector (REVISED)
                    if 'median_op_margin' in sector_grouped.columns:
                        st.markdown("**Median Operating Margin by Sector**")
                        sector_mar_plot = sector_grouped.dropna(subset=['median_op_margin']).nlargest(15, 'median_op_margin')
                        if not sector_mar_plot.empty:
                            # Convert nace2 to string for categorical plotting
                            sector_mar_plot['nace2_str'] = sector_mar_plot['nace2'].astype(str)
                            fig_sec_mar = px.bar(sector_mar_plot,
                                                 x='median_op_margin',
                                                 y='nace2_str', # Use string version
                                                 orientation='h',
                                                 title="Top 15 Sectors by Median Operating Margin",
                                                 labels={'median_op_margin': 'Median Operating Margin', 'nace2_str': 'Sector (NACE2)'},
                                                 hover_data=['company_count', 'nace2'],
                                                 height=plot_height,
                                                 text='median_op_margin' # Add text label
                                                 )
                            # Format the text label as percentage
                            fig_sec_mar.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                            fig_sec_mar.update_layout(
                                yaxis={'categoryorder':'total ascending', 'type': 'category'}, # Explicitly set category type
                                xaxis_tickformat='.1%', # Format x-axis as percentage
                                uniformtext_minsize=8, uniformtext_mode='hide'
                                )
                            st.plotly_chart(fig_sec_mar, use_container_width=True)
                        else: st.info("No valid sector Operating Margin data to display.")
                    else: st.info("Operating Margin data not available.")

                # --- Data Table ---
                st.subheader("Sector Data Table")
                # ... (Dataframe display code remains the same) ...
                # Check if there's any meaningful data to display beyond just sector code and count
                meaningful_cols = [col for col in sector_grouped.columns if col not in ['nace2', 'company_count']]
                if sector_grouped[meaningful_cols].notna().any(axis=None): # Check if *any* non-NaN value exists in data columns
                    format_dict = {}
                    if 'total_rd' in sector_grouped: format_dict['total_rd'] = '{:,.0f}'
                    if 'total_ns' in sector_grouped: format_dict['total_ns'] = '{:,.0f}'
                    if 'median_rd_intensity' in sector_grouped: format_dict['median_rd_intensity'] = '{:.2%}'
                    if 'avg_rd_intensity' in sector_grouped: format_dict['avg_rd_intensity'] = '{:.2%}'
                    if 'median_op_margin' in sector_grouped: format_dict['median_op_margin'] = '{:.2%}'
                    if 'total_ip5' in sector_grouped: format_dict['total_ip5'] = '{:,.0f}'

                    # Sort by total_rd if available, otherwise by count
                    sort_col = 'total_rd' if 'total_rd' in sector_grouped.columns else 'company_count'
                    st.dataframe(
                        sector_grouped.sort_values(by=sort_col, ascending=False)
                        .reset_index(drop=True)
                        .style.format(format_dict, na_rep='N/A') # Ensure N/A display
                    )
                else:
                    st.info("No aggregated sector data (like totals or medians) available to display in the table for the current filters.")

            else: # Handle if sector_grouped was empty initially (e.g., df was empty but had nace2 column)
                st.info("No sector data available to aggregate for the current selection.")
        else: # Handle if agg_dict only had 'company_count'
            st.info("Insufficient data columns available for meaningful sector aggregation (need RD, NS, etc.).")


# --- Tab 3: IP vs Financials ---
with tab_ip:
    st.subheader(f"IP Activity vs. Financial Performance ({sel_year})")
    ip_col = 'ip5_total'
    fin_cols = ['rd_intensity', 'op_margin', 'ns', 'rd']
    required_cols = [ip_col] + fin_cols + ['company_name', 'nace2']

    if df.empty:
         st.info("No data to display for the selected filters in the IP vs Financials tab.")
    elif not all(col in df.columns for col in required_cols):
         missing_required = [col for col in required_cols if col not in df.columns]
         st.warning(f"Cannot generate IP vs. Financials analysis. Missing required columns: {', '.join(missing_required)}")
    else:
        scatter_df = df.dropna(subset=[ip_col, 'rd_intensity', 'op_margin', 'ns']).copy()
        scatter_df['ns_size'] = scatter_df['ns'].fillna(0).clip(lower=0) # Ensure non-negative size

        # Filter out rows where essential plot columns are still NaN after initial dropna
        scatter_df = scatter_df.dropna(subset=[ip_col, 'rd_intensity', 'op_margin', 'ns_size'])

        if not scatter_df.empty:
            col1, col2 = st.columns(2)
            hover_list = ['rd', 'ns', 'op_margin', 'rd_intensity', ip_col]
            hover_data_subset = [c for c in hover_list if c in scatter_df.columns]

            # --- R&D Intensity vs IP5 ---
            with col1:
                st.markdown("**R&D Intensity vs. IP5 Patent Applications**")
                max_ip = scatter_df[ip_col].max()
                median_ip = scatter_df[ip_col].median()
                use_log_x = median_ip > 0 and pd.notna(max_ip) and pd.notna(median_ip) and max_ip > 10 * median_ip

                # Define Y-axis range (e.g., 0% to 100%)
                y_range_intensity = [0, 1] # Adjust if needed, e.g., [0, 0.5] for 0-50%

                fig_ip_int = px.scatter(scatter_df, x=ip_col, y='rd_intensity',
                                        size='ns_size', color='nace2',
                                        hover_name='company_name', hover_data=hover_data_subset,
                                        log_x=use_log_x,
                                        title=f"R&D Intensity (Focus: {y_range_intensity[0]:.0%} to {y_range_intensity[1]:.0%}) vs. IP5 Patents",
                                        labels={ip_col: f"IP5 Patent Applications {'(log scale)' if use_log_x else ''}",
                                                'rd_intensity': 'R&D Intensity', 'nace2': 'Sector'})

                fig_ip_int.update_layout(
                    yaxis_range=y_range_intensity, # Apply the range
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig_ip_int, use_container_width=True)
                st.caption(f"Y-axis limited to {y_range_intensity[0]:.0%} - {y_range_intensity[1]:.0%} R&D Intensity to focus on the main data cluster. Bubble size = Net Sales.")

            # --- Operating Margin vs IP5 ---
            with col2:
                st.markdown("**Operating Margin vs. IP5 Patent Applications**")
                # Define Y-axis range (e.g., -50% to +50%)
                y_range_margin = [-0.5, 0.5] # Adjust if needed, e.g., [-0.2, 0.3]

                fig_ip_mar = px.scatter(scatter_df, x=ip_col, y='op_margin',
                                        size='ns_size', color='nace2',
                                        hover_name='company_name', hover_data=hover_data_subset,
                                        log_x=use_log_x, # Use same log scale condition
                                        title=f"Operating Margin (Focus: {y_range_margin[0]:.0%} to {y_range_margin[1]:.0%}) vs. IP5 Patents",
                                        labels={ip_col: f"IP5 Patent Applications {'(log scale)' if use_log_x else ''}",
                                                'op_margin': 'Operating Margin', 'nace2': 'Sector'})

                fig_ip_mar.update_layout(
                    yaxis_range=y_range_margin, # Apply the range
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig_ip_mar, use_container_width=True)
                st.caption(f"Y-axis limited to {y_range_margin[0]:.0%} - {y_range_margin[1]:.0%} Operating Margin to focus on the main data cluster. Bubble size = Net Sales.")
        else:
            st.info("Insufficient valid data for IP vs. Financial scatter plots after filtering.")

# --- Tab 4: Growth Analysis ---
with tab_growth:
    st.subheader(f"Growth Analysis: {sel_year} vs. 2015")
    start_year = 2015

    if start_year == sel_year:
        st.info(f"Selected year is the baseline year ({start_year}). Cannot calculate growth.")
    elif start_year not in years:
         st.warning(f"Baseline year {start_year} not found in the dataset. Cannot perform growth analysis.")
    else:
        # (Filtering logic remains the same)
        mask_start = pd.Series(True, index=raw_df.index)
        if start_year and 'year' in raw_df.columns: mask_start &= (raw_df["year"] == start_year)
        if sel_countries and 'ctry_code' in raw_df.columns: mask_start &= raw_df["ctry_code"].isin(sel_countries)
        if sel_sectors and 'nace2' in raw_df.columns: mask_start &= raw_df["nace2"].isin(sel_sectors)
        df_start_year_filtered = raw_df.loc[mask_start].copy()

        if df_start_year_filtered.empty:
            st.warning(f"No data for selected filters in baseline year ({start_year}). Cannot calculate growth.")
        else:
            if 'company_name' not in df_start_year_filtered.columns: df_start_year_filtered['company_name'] = 'Company ' + df_start_year_filtered.index.astype(str)
            df_start = calculate_metrics(df_start_year_filtered)

            id_col = 'company_name'
            if id_col not in df.columns or id_col not in df_start.columns:
                 st.error(f"Company identifier '{id_col}' not found in both years' data.")
            elif df.empty or df_start.empty:
                 st.warning("No data for comparison in start or selected year after filtering.")
            else:
                cols_for_growth = ['rd', 'ns', 'op', 'emp', 'ip5_total', 'tm_total']
                growth_cols = [col for col in cols_for_growth if col in df.columns and col in df_start.columns]

                if not growth_cols:
                     st.warning("No common columns found between years for growth calculation.")
                else:
                    df_growth = pd.merge(
                        df_start[[id_col] + growth_cols], df[[id_col] + growth_cols],
                        on=id_col, suffixes=(f'_{start_year}', f'_{sel_year}'), how='inner'
                    )

                    if df_growth.empty:
                        st.warning(f"No matching companies found between {start_year} and {sel_year} with current filters.")
                    else:
                        # (Growth calculation logic remains the same)
                        for col in growth_cols:
                            col_start = f'{col}_{start_year}'; col_end = f'{col}_{sel_year}'; growth_col_name = f'{col}_growth_pct'
                            start_values = df_growth[col_start]; end_values = df_growth[col_end]
                            df_growth[growth_col_name] = np.nan
                            valid_mask = (start_values > 0) & start_values.notna() & end_values.notna()
                            df_growth.loc[valid_mask, growth_col_name] = (end_values[valid_mask] - start_values[valid_mask]) / start_values[valid_mask]
                            inf_mask = (start_values <= 0) & start_values.notna() & (end_values > 0) & end_values.notna()
                            df_growth.loc[inf_mask, growth_col_name] = np.inf
                            neg_mask = (start_values > 0) & start_values.notna() & (end_values <= 0) & end_values.notna()
                            df_growth.loc[neg_mask, growth_col_name] = -1.0

                        st.success(f"Growth calculated for **{len(df_growth)}** companies found in both {start_year} and {sel_year}.")

                        # --- Display Growth KPIs ---
                        st.subheader(f"Overall Median Growth ({start_year} to {sel_year})")
                        # Only show KPIs for calculated growth columns
                        growth_cols_calculated = [col for col in growth_cols if f'{col}_growth_pct' in df_growth.columns]
                        growth_kpi_cols = st.columns(len(growth_cols_calculated))
                        for i, col in enumerate(growth_cols_calculated):
                            median_growth = df_growth[f'{col}_growth_pct'].replace([np.inf, -np.inf], np.nan).median()
                            growth_kpi_cols[i].metric(
                                f"Median {col.upper()} Growth",
                                f"{median_growth:.1%}" if pd.notna(median_growth) else "N/A",
                                help=f"Median % change in {col.upper()} from {start_year} to {sel_year} for matched companies."
                            )

                        # --- Visualize Growth ---
                        # Use Violin Plot instead of Box Plot
                        if 'rd_growth_pct' in df_growth.columns and 'nace2' in df.columns:
                             df_growth_viz = pd.merge(df_growth[[id_col, 'rd_growth_pct']], df[[id_col, 'nace2']], on=id_col, how='left')
                             # Use a reasonable range, excluding Inf/-Inf and NaN for visualization
                             viz_data_growth = df_growth_viz.replace([np.inf, -np.inf], np.nan).dropna(subset=['rd_growth_pct', 'nace2'])
                             growth_range = [-1, 2] # -100% to +200%

                             if not viz_data_growth.empty:
                                 # Filter for the plot range *before* getting top sectors
                                 viz_data_growth_filtered = viz_data_growth[
                                     (viz_data_growth['rd_growth_pct'] >= growth_range[0]) &
                                     (viz_data_growth['rd_growth_pct'] <= growth_range[1])
                                 ]

                                 if not viz_data_growth_filtered.empty:
                                     st.markdown(f"**Distribution of R&D Growth (%) by Sector ({growth_range[0]:.0%} to {growth_range[1]:.0%})**")
                                     top_sectors_growth = viz_data_growth_filtered['nace2'].value_counts().nlargest(15).index
                                     viz_data_growth_top = viz_data_growth_filtered[viz_data_growth_filtered['nace2'].isin(top_sectors_growth)]

                                     # Use Violin Plot
                                     fig_growth_violin = px.violin(
                                         viz_data_growth_top, x='nace2', y='rd_growth_pct',
                                         title=f"R&D Growth Distribution ({start_year}-{sel_year}) by Top 15 Sectors",
                                         labels={'nace2': 'Sector (NACE2 - Top 15 by Company Count)', 'rd_growth_pct': 'R&D Growth (%)'},
                                         points=False, # Can add 'all', False, or 'outliers'
                                         box=True, # Show box plot inside violin
                                         hover_data=['company_name'] if id_col == 'company_name' else None # Add hover if useful
                                     )
                                     fig_growth_violin.update_layout(
                                         yaxis_tickformat='.0%',
                                         yaxis_range=growth_range, # Explicitly set range
                                         xaxis_title="Sector (Top 15 by Company Count)"
                                         )
                                     st.plotly_chart(fig_growth_violin, use_container_width=True)
                                     st.caption(f"Y-axis limited to {growth_range[0]:.0%} - {growth_range[1]:.0%} R&D Growth. Shows distribution shape (violin) and quartiles (box).")
                                 else:
                                     st.info(f"No R&D growth data within the range {growth_range[0]:.0%} to {growth_range[1]:.0%} to display.")
                             else:
                                 st.info("No valid R&D growth data to display.")
                        else:
                             st.info("R&D growth or Sector information not available for visualization.")

                        st.subheader("Company Growth Data Sample")
                        st.dataframe(df_growth.head().style.format("{:.1%}", subset=[col for col in df_growth.columns if '_growth_pct' in col], na_rep='N/A')
                                                          .format("{:,.0f}", subset=[col for col in df_growth.columns if '_growth_pct' not in col and col != id_col], na_rep='N/A'))


# --- Tab 5: External Report Summary ---
# (Keep as is from previous version, already improved)
with tab_report:
    st.subheader("📘 Insights from JRC–OECD Report (2017 Edition)")
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
        pie_data_portfolio = pd.DataFrame({
            "IP Type": ["Patents related to ICT", "Designs related to ICT", "Trademarks related to ICT"],
            "Approx Portfolio Share": [45, 25, 25]
        })
        pie_chart_portfolio = px.pie(pie_data_portfolio, names="IP Type", values="Approx Portfolio Share",
                           title="Approx. Share of ICT within<br>Top Firms' Own IP Portfolio", height=250)
        pie_chart_portfolio.update_traces(textinfo='percent+label', insidetextorientation='radial')
        pie_chart_portfolio.update_layout(margin=dict(t=60, b=10, l=0, r=0), legend_title_text='IP Type')
        st.plotly_chart(pie_chart_portfolio, use_container_width=True)

    st.info("Reference: Daiko T., Dernis H., Dosso M., Gkotsis P., Squicciarini M., Vezzani A. (2017). *World Corporate Top R&D Investors: Industrial Property Strategies in the Digital Economy*. A JRC and OECD common report. Luxembourg: Publications Office of the European Union. EUR 28656 EN.")

# --- NEW Tab 6: Regression Analysis ---
with tab_regression:
    st.header("📉 Regression Analysis (Illustrative)")
    st.markdown(f"""
    This section presents simple Ordinary Least Squares (OLS) regression results
    based on the **cross-sectional data filtered for the selected year ({sel_year})**.
    These models explore potential correlations suggested by the hypotheses in the 'Analysis Framework' tab.

    **Important Caveats:**
    *   **Correlation, Not Causation:** These models show associations, *not* causal relationships.
    *   **Cross-Sectional:** Analysis is only for the selected year, not panel data which would be needed for more robust analysis of lags and fixed effects.
    *   **Omitted Variables:** Many factors influencing profitability are not included (e.g., market structure, firm strategy, macroeconomic conditions).
    *   **Endogeneity:** R&D/IP might influence profits, but profits might also influence R&D/IP investment (simultaneity).
    *   **Illustrative Only:** Results are intended for exploratory purposes within this dashboard's context.
    """)
    st.info("Models are run only if sufficient data (non-missing values for all variables in the model) exists after applying filters.")

    if df.empty:
        st.warning("No data available for regression after filtering.")
    else:
        # --- Define Models ---
        models_to_run = {
            "Model 1: Profit Margin vs. R&D Intensity": {
                "formula": "op_margin ~ rd_intensity",
                "required_cols": ["op_margin", "rd_intensity"]
            },
            "Model 2: Profit Margin vs. Patent Count": {
                "formula": "op_margin ~ ip5_total",
                "required_cols": ["op_margin", "ip5_total"]
            },
            "Model 3: Profit Margin vs. R&D Intensity & Patents": {
                "formula": "op_margin ~ rd_intensity + ip5_total",
                "required_cols": ["op_margin", "rd_intensity", "ip5_total"]
            },
            "Model 4: Model 3 + Size Control (Log(Employees))": {
                "formula": "op_margin ~ rd_intensity + ip5_total + np.log1p(emp)", # log1p handles potential zeros in emp
                "required_cols": ["op_margin", "rd_intensity", "ip5_total", "emp"]
            }
        }

        results_html = {} # Store results

        for model_name, model_info in models_to_run.items():
            formula = model_info["formula"]
            required = model_info["required_cols"]

            # Check if required columns exist
            if not all(col in df.columns for col in required):
                results_html[model_name] = f"<p><i>Skipped: Missing required columns ({', '.join([c for c in required if c not in df.columns])}).</i></p>"
                continue

            # Prepare data for this specific model (drop rows with NaNs in needed columns)
            df_model = df[required].dropna()

            # Check if enough data remains (need more observations than predictors)
            num_predictors = formula.count('+') + 1 # Count terms on RHS
            if len(df_model) <= num_predictors:
                results_html[model_name] = f"<p><i>Skipped: Insufficient non-missing data ({len(df_model)} observations) for required columns ({', '.join(required)}).</i></p>"
                continue

            # Run the OLS regression
            try:
                with warnings.catch_warnings(): # Suppress potential multicollinearity or other warnings
                    warnings.simplefilter("ignore")
                    model = smf.ols(formula=formula, data=df_model).fit()
                results_html[model_name] = model.summary().as_html() # Get summary table as HTML
            except Exception as e:
                results_html[model_name] = f"<p><i>Error running model: {e}</i></p>"

        # --- Display Results ---
        st.subheader("OLS Regression Results")
        for model_name, summary_html in results_html.items():
            st.markdown(f"--- \n #### {model_name}")
            # Display the formula used
            if "formula" in models_to_run.get(model_name, {}):
                 st.code(f"Formula: {models_to_run[model_name]['formula']}", language="python")
            # Display the results table (or error message)
            st.markdown(summary_html, unsafe_allow_html=True)
