import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import dash_bootstrap_components as dbc


# Lecture du cache: données de prix SPOT réalisé France
# Téléchargeable à: https://ember-energy.org/data/european-wholesale-electricity-price-data/
df = pd.read_csv("data/France - extrait.csv")
df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
df.sort_values(by="Datetime (UTC)", inplace=True)
min_date: pd.Timestamp = df["Datetime (UTC)"].min()
max_date: pd.Timestamp = df["Datetime (UTC)"].max()

spot_layout = html.Div(
    style={
        "backgroundColor": "lightblue",
        "borderRadius": "10px",
        "padding": "20px",
        "margin": "20px",
    },
    children=[
        html.H2("Prix SPOT (France)"),
        html.Div(
            [
                dbc.Button(
                    "Dèrniere semaine", 
                    id="derniere-semaine-button", 
                    n_clicks=0,
                    color = 'primary',
                    className = "me-2"
                ),
                dbc.Button("Dernier mois", 
                           id="dernier-mois-button", 
                           n_clicks=0,
                           color = "secondary"
                ),
            ]
        ),
        dcc.DatePickerRange(
            id="date-picker-range",
            min_date_allowed=min_date,
            max_date_allowed=max_date,
            start_date=max_date - pd.DateOffset(weeks=1),
            end_date=max_date,
            display_format="YYYY-MM-DD",
            clearable=False,
        ),
        dcc.Graph(id="price-time-series"),
    ],
)


@callback(
    Output("price-time-series", "figure"),
    Input("date-picker-range", "start_date"),
    Input("date-picker-range", "end_date"),
)
def update_graph(start_date, end_date):
    filtered_df = df[
        (df["Datetime (UTC)"] >= start_date) & (df["Datetime (UTC)"] <= end_date)
    ]

    # La figure principale
    fig = px.line(
        filtered_df,
        x="Datetime (UTC)",
        y="Price (EUR/MWhe)",
        title="Prix SPOT",
    )

    # Ajoute un background rouge semi-transparent
    negative_prices = filtered_df[filtered_df["Price (EUR/MWhe)"] < 0]
    for i in range(len(negative_prices)):
        start_time = negative_prices["Datetime (UTC)"].iloc[i]
        fig.add_shape(
            type="rect",
            x0=start_time,
            x1=start_time + pd.Timedelta(hours=1),
            y0=filtered_df["Price (EUR/MWhe)"].min(),
            y1=filtered_df["Price (EUR/MWhe)"].max(),
            fillcolor="red",
            opacity=0.2,
            line=dict(width=0),
            layer="below",
        )

    return fig


@callback(
    Output("date-picker-range", "start_date"),
    Output("date-picker-range", "end_date"),
    Input("derniere-semaine-button", "n_clicks"),
    Input("dernier-mois-button", "n_clicks"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
)
def update_date_range(n_clicks_dernier_mois, n_clicks_six_mois, start_date, end_date):
    ctx = dash.callback_context
    changed_id = ctx.triggered[0]["prop_id"]
    if "derniere-semaine-button" in changed_id:
        return max_date - pd.DateOffset(weeks=1), max_date
    elif "dernier-mois-button" in changed_id:
        return max_date - pd.DateOffset(months=1), max_date
    else:
        return start_date, end_date
