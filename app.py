import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Global Diabetes – Thailand in World Context",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path("NCD_RisC_Lancet_2024_Diabetes_age_standardised_countries.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_global_diabetes():
    df = pd.read_csv(DATA_PATH)

    cols = {
        "country": "Country/Region/World",
        "iso": "ISO",
        "sex": "Sex",
        "year": "Year",
        "prev": "Prevalence of diabetes (18+ years)",
        "treated": "Proportion of people with diabetes who were treated (30+ years)",
    }

    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nAvailable: {list(df.columns)}")

    return df, cols


df, cols = load_global_diabetes()

country_col = cols["country"]
iso_col = cols["iso"]
sex_col = cols["sex"]
year_col = cols["year"]
prev_col = cols["prev"]
treated_col = cols["treated"]

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

# Sex filter
unique_sex = sorted(df[sex_col].dropna().unique().tolist())
sex_options = ["All"] + unique_sex
sex_selected = st.sidebar.selectbox("Sex", sex_options, index=0)

# Year selector
year_min = int(df[year_col].min())
year_max = int(df[year_col].max())
year_selected = st.sidebar.slider(
    "Year",
    min_value=year_min,
    max_value=year_max,
    value=year_max,
    step=1,
)

# Map metric
map_metric = st.sidebar.radio(
    "Map metric",
    options=["Prevalence", "Treatment"],
    index=0,
)

# Button-style prevalence bands
st.sidebar.markdown("### Prevalence band")
prev_band = st.sidebar.radio(
    "Prevalence filter",
    options=[
        "All",
        "High (≥15%)",
        "Medium (8–15%)",
        "Low (<8%)",
    ],
    index=0,
)

# Button-style treatment bands
st.sidebar.markdown("### Treatment band")
treat_band = st.sidebar.radio(
    "Treatment filter",
    options=[
        "All",
        "High (≥60%)",
        "Medium (30–60%)",
        "Low (<30%)",
    ],
    index=0,
)

# Country to highlight on scatter
highlight_country = st.sidebar.selectbox(
    "Extra country highlight (scatter)",
    ["None"] + sorted(df[country_col].unique().tolist()),
    index=0,
)

# Country for trend view
all_countries_sorted = sorted(df[country_col].unique().tolist())
default_idx = all_countries_sorted.index("Thailand") if "Thailand" in all_countries_sorted else 0
trend_country = st.sidebar.selectbox(
    "Country for time trend",
    all_countries_sorted,
    index=default_idx,
)

# -----------------------------
# DATA FILTERING FOR YEAR-BASED VIEWS (MAP/SCATTER/RANKINGS)
# -----------------------------
if sex_selected == "All":
    df_sex_year = (
        df.groupby([country_col, iso_col, year_col], as_index=False)
        .mean(numeric_only=True)
    )
    df_sex_year[sex_col] = "All"
else:
    df_sex_year = df[df[sex_col] == sex_selected].copy()

df_year = df_sex_year[df_sex_year[year_col] == year_selected].copy()

if df_year.empty:
    st.error("No data for this combination of filters.")
    st.stop()

# Convert to percent
df_year[prev_col] = df_year[prev_col] * 100
df_year[treated_col] = df_year[treated_col] * 100

# Apply prevalence band
if prev_band == "High (≥15%)":
    df_year = df_year[df_year[prev_col] >= 15]
elif prev_band == "Medium (8–15%)":
    df_year = df_year[(df_year[prev_col] >= 8) & (df_year[prev_col] < 15)]
elif prev_band == "Low (<8%)":
    df_year = df_year[df_year[prev_col] < 8]

# Apply treatment band
if treat_band == "High (≥60%)":
    df_year = df_year[df_year[treated_col] >= 60]
elif treat_band == "Medium (30–60%)":
    df_year = df_year[(df_year[treated_col] >= 30) & (df_year[treated_col] < 60)]
elif treat_band == "Low (<30%)":
    df_year = df_year[df_year[treated_col] < 30]

if df_year.empty:
    st.error("No countries match these band filters – try relaxing them.")
    st.stop()

# Map metric + colors
if map_metric == "Prevalence":
    map_col = prev_col
    map_title_metric = "Age-standardised diabetes prevalence (%)"
    map_scale = "Reds"
else:
    map_col = treated_col
    map_title_metric = "Age-standardised treated (% of people with diabetes)"
    map_scale = "Blues"  # treatment = blue

# -----------------------------
# HEADER
# -----------------------------
st.title("Global Diabetes – Thailand’s Position in the World")

st.markdown(
    f"""
Using **NCD-RisC / Lancet 2024** age-standardised diabetes data.

- Map metric: **{map_title_metric}**  
- Sex: **{sex_selected}**  
- Year: **{year_selected}**  
- Filters: **{prev_band} prevalence**, **{treat_band} treatment**
"""
)

# -----------------------------
# THAILAND KPIs
# -----------------------------
th_mask = df_year[country_col].str.lower().eq("thailand")
has_thailand = th_mask.any()

col1, col2, col3 = st.columns(3)

if has_thailand:
    th_row = df_year[th_mask].iloc[0]
    th_prev = th_row[prev_col]
    th_treated = th_row[treated_col]

    df_year["prev_rank"] = df_year[prev_col].rank(ascending=False, method="min")
    df_year["treated_rank"] = df_year[treated_col].rank(ascending=False, method="min")
    th_prev_rank = int(df_year.loc[th_mask, "prev_rank"].iloc[0])
    th_treat_rank = int(df_year.loc[th_mask, "treated_rank"].iloc[0])
    n_countries = len(df_year)

    with col1:
        st.metric(f"Thailand prevalence in {year_selected}", f"{th_prev:.1f}%")
    with col2:
        st.metric(f"Thailand treated in {year_selected}", f"{th_treated:.1f}%")
    with col3:
        st.write(
            f"**Prevalence rank (within filtered set):** {th_prev_rank}/{n_countries}  \n"
            f"**Treatment rank (within filtered set):** {th_treat_rank}/{n_countries}"
        )
else:
    col1.write("Thailand not in current filtered set (bands may be too restrictive).")

st.markdown("---")

# -----------------------------
# SECTION 1 – WORLD MAP
# -----------------------------
st.subheader("1. World Map – Diabetes Prevalence or Treatment")

fig_map = px.choropleth(
    df_year,
    locations=iso_col,
    locationmode="ISO-3",
    color=map_col,
    hover_name=country_col,
    hover_data={prev_col: ":.1f", treated_col: ":.1f"},
    color_continuous_scale=map_scale,
    labels={prev_col: "Prevalence (%)", treated_col: "Treated (%)"},
    title=f"{map_title_metric} in {year_selected} ({sex_selected})",
)
fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_map, use_container_width=True)

st.caption(
    "Use the prevalence and treatment **band buttons** in the sidebar to focus on high / medium / low burden and coverage."
)

# -----------------------------
# SECTION 2 – SCATTER
# -----------------------------
st.subheader("2. Global scatter – prevalence vs treatment (care gap)")

scatter_df = df_year.copy()


def label_country(c):
    c_low = c.lower()
    if c_low == "thailand":
        return "Thailand"
    if highlight_country != "None" and c == highlight_country:
        return highlight_country
    return "Other"


scatter_df["Highlight"] = scatter_df[country_col].apply(label_country)

fig_scatter = px.scatter(
    scatter_df,
    x=prev_col,
    y=treated_col,
    color="Highlight",
    hover_name=country_col,
    labels={prev_col: "Prevalence (%)", treated_col: "Treated (% of people with diabetes)"},
    title=f"Prevalence vs treatment coverage in {year_selected}",
)
fig_scatter.update_traces(marker=dict(size=9))
st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "Top-right = high burden & high treatment. Countries far below the diagonal (high prevalence, low treatment) have large unmet need."
)

# -----------------------------
# SECTION 3 – RANKINGS
# -----------------------------
st.subheader("3. Country rankings – top & bottom 15")

col_r1, col_r2 = st.columns(2)
top_n = 15

metric_choice = st.radio(
    "Ranking metric",
    options=["Prevalence", "Treatment"],
    index=0,
    horizontal=True,
)

if metric_choice == "Prevalence":
    metric_col = prev_col
    metric_label = "Prevalence (%)"
else:
    metric_col = treated_col
    metric_label = "Treated (%)"

with col_r1:
    top_df = df_year.sort_values(metric_col, ascending=False).head(top_n)
    fig_top = px.bar(
        top_df,
        x=metric_col,
        y=country_col,
        orientation="h",
        labels={metric_col: metric_label},
        title=f"Top {top_n} – highest {metric_label}",
    )
    fig_top.update_layout(yaxis_categoryorder="total ascending")
    st.plotly_chart(fig_top, use_container_width=True)

with col_r2:
    bottom_df = df_year.sort_values(metric_col, ascending=True).head(top_n)
    fig_bottom = px.bar(
        bottom_df,
        x=metric_col,
        y=country_col,
        orientation="h",
        labels={metric_col: metric_label},
        title=f"Bottom {top_n} – lowest {metric_label}",
    )
    fig_bottom.update_layout(yaxis_categoryorder="total ascending")
    st.plotly_chart(fig_bottom, use_container_width=True)

st.caption(
    "These rankings show where diabetes burden or treatment coverage is highest and lowest within the filtered group."
)

# -----------------------------
# SECTION 4 – COUNTRY FOCUS TREND
# -----------------------------
st.subheader("4. Country focus – trend over time")

st.markdown(
    f"Time-series of **{trend_country}** showing how diabetes burden or treatment "
    f"has changed over time (same sex filter: **{sex_selected}**)."
)

# build trend dataset (sex filter only, all years)
if sex_selected == "All":
    df_sex_all = (
        df.groupby([country_col, iso_col, year_col], as_index=False)
        .mean(numeric_only=True)
    )
    df_sex_all[sex_col] = "All"
else:
    df_sex_all = df[df[sex_col] == sex_selected].copy()

df_trend = df_sex_all[df_sex_all[country_col] == trend_country].copy()
df_trend.sort_values(year_col, inplace=True)

if df_trend.empty:
    st.warning("No data for this country with the current sex filter.")
else:
    df_trend[prev_col] = df_trend[prev_col] * 100
    df_trend[treated_col] = df_trend[treated_col] * 100

    metric_trend_choice = st.radio(
        "Trend metric",
        options=["Prevalence", "Treatment"],
        index=0,
        horizontal=True,
    )

    if metric_trend_choice == "Prevalence":
        y_col = prev_col
        y_label = "Prevalence (%)"
        title_suffix = "diabetes prevalence"
    else:
        y_col = treated_col
        y_label = "Treated (% of people with diabetes)"
        title_suffix = "treatment coverage"

    fig_trend = px.line(
        df_trend,
        x=year_col,
        y=y_col,
        markers=True,
        labels={year_col: "Year", y_col: y_label},
        title=f"{trend_country} – {title_suffix} over time",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
