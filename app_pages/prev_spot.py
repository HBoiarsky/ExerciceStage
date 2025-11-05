from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import os
import pandas as pd
import pickle
import base64
import io

import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
prev_spot_layout = html.Div(
    style={
        "backgroundColor": "lightblue",
        "borderRadius": "10px",
        "padding": "20px",
        "margin": "20px",
    },
    children=[
        html.H2("Prévisions de prix SPOT"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Glisser-déposer ou sélectionner un fichier CSV à importer"]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="output-data-upload"),
        html.P(
            [
                "Veuillez importer un fichier CSV contenant les données historiques (téléchargé depuis ",
                html.A(
                    "eco2mix",
                    href="https://www.rte-france.com/eco2mix/telecharger-les-indicateurs",
                    target="_blank",
                    rel="noopener noreferrer",
                ),
                ").",
            ]
        ),
        html.H3("Sélectionner un modèle:"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": f, "value": f} for f in model_files],
            placeholder="Sélectionner un modèle",
        ),
        dbc.Button(
            "Lancer les prévisions", 
            id="run-forecasts-button", 
            n_clicks=0,
            color="success",
            className="mt-3"
        ),
        html.Div(id="forecast-output"),
    ],
)


@callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def upload_input_file(contents, filename):
    if contents is None:
        return html.Div()
    try:
        # Les fichiers eCO2mix_RTE_*.xls sont en réalité des .csv séparés par des \t
        # Ils sont encodés en latin1 (présence d'accents),
        # et la dernière ligne est un message d'avertissement, à supprimer
        _, content_string = contents.split("base64,")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(
            io.StringIO(decoded.decode("latin1")), sep="\t", index_col=False
        ).iloc[:-1]

        # Si l'import a réussi, affiche le nom du fichier et ses premières lignes
        return html.Div([html.H5(filename), html.Hr(), html.Div(df.head().to_string())])
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])


@callback(
    Output("forecast-output", "children"),
    Input("run-forecasts-button", "n_clicks"),
    State("model-dropdown", "value"),
    State("upload-data", "contents"),
)
def run_forecasts(n_clicks, model_filename, contents):
    if not (n_clicks > 0 and model_filename and contents):
        return html.Div()

    try:
        _, content_string = contents.split("base64,")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(
            io.StringIO(decoded.decode("latin1")), sep="\t", index_col=False
        ).iloc[:-1]

        with open(f"models/{model_filename}", "rb") as file:
            model = pickle.load(file)
        try:
            # PRÉPARATION DES DONNÉES 
            # Sélection des colonnes nécessaires au modèle et nettoyage
            # - Garde uniquement les features utilisées lors de l'entraînement
            # - Remplace les valeurs "ND" par NaN
            # - Supprime les lignes avec des valeurs manquantes
            df_clean = df[model.feature_names_in_].replace('ND', float('nan')).dropna()

            # Le modèle prédit le prix pour chaque heure sur base des données éco2mix
            previsions_prix_spot = model.predict(df_clean)

            # CHARGEMENT DES PRIX SPOT RÉELS 
            # Lecture des prix historiques
            df_spot = pd.read_csv("data/France - extrait.csv")
            # Conversion de la colonne datetime en format pandas pour le matching
            df_spot["Datetime (UTC)"] = pd.to_datetime(df_spot["Datetime (UTC)"])

            # PRÉPARATION DES DATES POUR LE MATCHING 
            df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Heures"])

            # CONSTRUCTION DU DATAFRAME RÉSULTATS 
            # Création d'un DataFrame contenant les prévisions et les données originales
            # On utilise les index de df_clean pour garder uniquement les lignes valides
            df_results = df.loc[df_clean.index].copy()
            # Ajout de la colonne des prévisions
            df_results["prevision"] = previsions_prix_spot

            # FUSION AVEC LES PRIX RÉELS 
            # Jointure left pour matcher les prévisions avec les prix réels historiques
            # left join : on garde toutes les prévisions même sans prix réel correspondant
            df_results = df_results.merge(
                df_spot[["Datetime (UTC)", "Price (EUR/MWhe)"]],
                left_on = "datetime",
                right_on = "Datetime (UTC)",
                how = "left"
            )

            # CALCUL DES MÉTRIQUES DE PERFORMANCE 
            # Création d'un masque pour identifier les lignes avec un prix réel disponible
            valid_mask = df_results["Price (EUR/MWhe)"].notna()
            if valid_mask.sum() > 0 :

                # Calcul de la MAE
                mae = mean_absolute_error(
                    df_results.loc[valid_mask, "Price (EUR/MWhe)"],
                    df_results.loc[valid_mask, "prevision"]
                )

                # Calcul de la RMSE 
                rmse = np.sqrt(mean_squared_error(
                    df_results.loc[valid_mask, "Price (EUR/MWhe)"],
                    df_results.loc[valid_mask, "prevision"]
                ))


                # CRÉATION DU GRAPHIQUE COMPARATIF
                fig = go.Figure()

                # Ajout de la courbe des prix réels (en bleu)
                fig.add_trace(go.Scatter(
                    x=df_results["datetime"],
                    y=df_results["Price (EUR/MWhe)"],
                    mode='lines',
                    name='Prix réels',
                    line=dict(color='blue', width=2)
                ))
                
                # Ajout de la courbe des prix prévisions (en rouge)
                fig.add_trace(go.Scatter(
                    x=df_results["datetime"],
                    y=df_results["prevision"],
                    mode='lines',
                    name='Prévisions',
                    line=dict(color='red', width=2, dash='dash')
                ))
                

                # Configuration du layout du graphique
                fig.update_layout(
                    title="Comparaison Prévisions vs Prix Réels",
                    xaxis_title="Date et Heure",
                    yaxis_title="Prix (EUR/MWhe)",
                    hovermode='x unified',
                    template="plotly_white"
                )

                # SAUVEGARDE DES PRÉVISIONS 
                # Export des prévisions en CSV 
                pd.DataFrame(previsions_prix_spot).to_csv("data/previsions.csv")

                # AFFICHAGE DES RÉSULTATS 
                return html.Div([
                    dbc.Alert([
                        html.H4("✓ Prévisions réalisées avec succès!", className="alert-heading"),
                        html.Hr(),
                        html.P([
                            html.Strong(f"Nombre de prévisions : "), f"{len(previsions_prix_spot)} heures",
                            html.Br(),
                            html.Strong(f"MAE (Erreur Absolue Moyenne) : "), f"{mae:.2f} EUR/MWhe",
                            html.Br(),
                            html.Strong(f"RMSE (Erreur Quadratique Moyenne) : "), f"{rmse:.2f} EUR/MWhe"
                        ])
                    ], color="success", className="mt-3"),
                    dcc.Graph(figure=fig, className="mt-3")
                ])
            
            # CAS 2 : Pas de données réelles  : affichage des prévisions uniquement
            else :
                # CRÉATION DU GRAPHIQUE (prévisions seules) 
                fig = go.Figure()

                # Ajout uniquement de la courbe des prévisions (en rouge)
                fig.add_trace(go.Scatter(
                    x=df_results["datetime"],
                    y=df_results["prevision"],
                    mode='lines',
                    name='Prévisions',
                    line=dict(color='red', width=2)
                ))
                
                # Configuration du layout du graphique
                fig.update_layout(
                    title="Prévisions de prix SPOT",
                    xaxis_title="Date et heure",
                    yaxis_title="Prix (EUR/MWhe)",
                    template="plotly_white"
                )
                
                # SAUVEGARDE DES PRÉVISIONS 
                pd.DataFrame(previsions_prix_spot).to_csv("data/previsions.csv")
                
                #  AFFICHAGE DES RÉSULTATS  
                return html.Div([
                    dbc.Alert([
                        html.H4("Prévisions réalisées avec succès", className="alert-heading"),
                        html.P(f"Nombre de prévisions : {len(previsions_prix_spot)} heures"),
                        html.P("Aucune donnée réelle correspondante trouvée pour comparaison", className="text-warning")
                    ], color="info", className="mt-3"),
                    dcc.Graph(figure=fig, className="mt-3")
                ])

        except Exception as e:
            # GESTION DES ERREURS D'EXÉCUTION 
            return dbc.Alert([
                html.H4("Erreur", className="alert-heading"),
                html.P(f"Erreur lors de l'exécution du modèle: {e}")
            ], color="danger", className="mt-3")
            

    except Exception as e:
        #  GESTION DES ERREURS DE CHARGEMENT 
        return dbc.Alert([
            html.H4("Erreur", className="alert-heading"),
            html.P(f"Erreur lors du chargement ou de l'exécution du modèle: {e}")
        ], color="danger", className="mt-3")
