#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["streamlit>=1.50,<2", "altair>=5,<7", "polars>=1,<2"]
# ///
# How to run: uv run --with streamlit --with altair --with polars streamlit run app.py
"""An executable example for the Week 13 deployment walkthrough."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
import streamlit as st

DATA_PATH: Final = Path(__file__).with_name("gapminder_2000_2020.csv")
INDICATORS: Final = {
    "Life expectancy (years)": "life_expectancy",
    "Fertility rate (children per woman)": "fertility_rate",
    "Child mortality (per 1,000 births)": "child_mortality",
}


@st.cache_data
def load_data() -> pl.DataFrame:
    """Read the bundled country statistics with explicit numeric types."""
    return pl.read_csv(
        DATA_PATH,
        schema_overrides={
            "year": pl.Int64,
            "population": pl.Float64,
            "income_per_capita": pl.Float64,
            "life_expectancy": pl.Float64,
            "fertility_rate": pl.Float64,
            "child_mortality": pl.Float64,
        },
    )


def main() -> None:
    """Render a native Streamlit dashboard with working year and region filters."""
    st.set_page_config(page_title="World Development Dashboard", layout="wide")
    data = load_data()
    years = sorted(data.get_column("year").unique().to_list())
    regions = sorted(data.get_column("region").unique().to_list())

    st.sidebar.header("Filters")
    year = st.sidebar.select_slider("Year", options=years, value=years[-1])
    chosen_regions = st.sidebar.multiselect("Region", regions, default=regions)
    indicator_label = st.sidebar.selectbox("Indicator", list(INDICATORS))
    indicator = INDICATORS[indicator_label]
    selected = data.filter(
        (pl.col("year") == year) & pl.col("region").is_in(chosen_regions),
    )

    st.title("World Development Dashboard")
    st.caption(
        f"Gapminder indicators, {year} · {len(chosen_regions)} of "
        f"{len(regions)} regions · {selected.height} countries",
    )
    if selected.is_empty():
        st.info("Select at least one region in the sidebar to display the charts.")
        st.stop()

    with st.container(horizontal=True):
        st.metric("Countries", selected.height, border=True)
        st.metric(
            "Population",
            f"{selected.get_column('population').sum() / 1e9:.2f} B",
            border=True,
        )
        st.metric(
            "Median income per capita",
            f"${selected.get_column('income_per_capita').median():.1f}k",
            border=True,
        )
        st.metric(
            "Median life expectancy",
            f"{selected.get_column('life_expectancy').median():.1f}",
            border=True,
        )

    st.subheader("Overview")
    chart_data = alt.Data(values=selected.to_dicts())
    scatter = (
        alt.Chart(chart_data)
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X(
                "income_per_capita:Q",
                title="Income per capita (thousand $, log scale)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(f"{indicator}:Q", title=indicator_label, scale=alt.Scale(zero=False)),
            color=alt.Color("region:N", title="Region"),
            size=alt.Size("population:Q", legend=None, scale=alt.Scale(range=[25, 1200])),
            tooltip=["country:N", "region:N", f"{indicator}:Q", "income_per_capita:Q"],
        )
        .properties(height=360, title=f"{indicator_label} vs. income, {year}")
        .interactive()
    )
    top = selected.drop_nulls(indicator).sort(indicator, descending=True).head(10)
    bars = (
        alt.Chart(alt.Data(values=top.to_dicts()))
        .mark_bar()
        .encode(
            x=alt.X(f"{indicator}:Q", title=indicator_label),
            y=alt.Y("country:N", title=None, sort="-x"),
            color=alt.Color("region:N", legend=None),
            tooltip=["country:N", f"{indicator}:Q"],
        )
        .properties(height=360, title=f"Top 10 countries, {year}")
    )
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(scatter, width="stretch")
    with right:
        st.altair_chart(bars, width="stretch")

    st.caption(
        "Deployment practice example · Gapminder class dataset, 2000–2020. "
        "This app was created for the walkthrough and is separate from the lecture's original app.",
    )


if __name__ == "__main__":
    main()
