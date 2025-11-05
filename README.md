Cette application Dash est composée de 3 onglets, et a été conçue comme exercice de recrutement pour un stage à EDF.

L'idée est d'utiliser les données de RTE (éco2mix) pour faire des prévisions de prix SPOT: pour une heure donnée, les valeurs des colonnes d'un fichier éco2mix doivent être utilisées pour prévoir le prix SPOT associé à l'heure en question. A noter que d'un point de vue "physique", cette prévision est sans intérêt, car les prix SPOT sont déterminés publiquement la veille pour le lendemain. Il s'agit simplement d'un exercice élémentaire, basé à 100% sur des données publiques et faciles d'accès, en lien avec le secteur de l'énergie. Si vous préparez d'autres modèles, il n'est donc pas nécessaire de viser à tout prix la performance de prévision.

Les 3 onglets sont:
- éco2mix 2024: onglet pour visualiser les données éco2mix sur toute l'année 2024. Les données doivent être stockées dans le dossier `data` de l'application.
- prix SPOT: onglet pour visualiser les prix SPOT. Les données doivent être stockées dans le dossier `data` de l'application.
- Prévision de prix SPOT: à partir d'un fichier eCO2mix_RTE_YYYY-MM-DD.xls (jour par jour) téléchargable sur éco2mix, et d'un modèle préalablement entraîné, faire une prévision et visualiser son résultat.

## Lien vers les documentations officielles utiles:
- [dash](https://dash.plotly.com/)
- [dash-bootstrap-components](https://www.dash-bootstrap-components.com/)
- [plotly](https://plotly.com/)
- [pandas](https://pandas.pydata.org/)


## Comment lancer cette application ?
1. Assurez-vous d'avoir les dépendances (cf requirements.txt) dans votre environnement python
2. Depuis un terminal, exécutez la commande suivante pour lancer l'application :
```
python main.py
```
4. Le message ci-dessous doit s'afficher dans la console.
Faites un Ctrl+Click sur (ou copiez-collez) le lien `http://127.[...]` pour ouvrir l'application dans votre navigateur par défaut:
```
Dash is running on http://127.0.0.1:8050/

* Serving Flask app 'main'
* Debug mode: on
```

## Configuration
Télécharger les données "En-cours annuel consolidé" d'[éCO2mix](https://www.rte-france.com/eco2mix/la-production-delectricite-par-filiere) sur [ce lien](https://www.rte-france.com/eco2mix/telecharger-les-indicateurs). \
Attention, le format du fichier téléchargé est un .csv (séparé par des tabulations), et pas un .xls comme l'extension le laisse penser... :)

Les prix SPOT de l'électricité peuvent être téléchargés [ici](https://ember-energy.org/data/european-wholesale-electricity-price-data/).
Choisir "hourly", puis récupérer le fichier "France.csv".

## Ajout de modèle de prévision de prix SPOT
A votre charge de créer un ou plusieurs modèles (simples, l'objectif n'est pas la performance) de prévision de prix SPOT pour les intégrer dans le dossier "models". Chacun de ces modèles doivent être des .pkl (voir [pickle](https://docs.python.org/3/library/pickle.html)) implémentant une méthode `predict()` (comme la quasi-totalité des modèles de sklearn). \
Le modèle doit pouvoir prendre comme input une dataframe dont les colonnes sont incluses dans celles des fichiers d'éco2mix, et avoir comme sortie le prix SPOT pour le créneau horaire correspondant.
Vous pouvez par exemple entraîner un modèle linéaire et une forêt aléatoire depuis les données "En-cours annuel consolidé" d'éco2mix et des prix SPOT depuis un notebook python, et enregistrer les modèles ainsi créés dans des `.pkl`.

## Attendu pour l'exercice de stage
Ajoutez une ou plusieurs fonctionnalités à cette application ! Quelques idées:
- onglet "Données éco2mix 2024": ajouter une ou plusieurs visualisations avec plotly, si possible avec des composants interactifs de dash (sélecteur de dates, ...)
- onglet "Prévision de prix SPOT": ajouter un formulaire pour construire un input à partir des données renvoyées par [l'API d'éco2mix](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/api/?disjunctive.nature).
- onglet "Prévision de prix SPOT": exploiter les prévisions faites par le modèle, par exemple en les affichant, les comparant au réalisé de prix SPOT dans `data/France.csv`, etc.
- tous les onglets: modifier les boutons pour utiliser `dbc.Button` à la place (ou autre améliorations de style)


# Fonctionnalités ajoutées

## 1. Amélioration des boutons (Bootstrap)
- Remplacement des boutons HTML par dbc.Button
- Ajout de couleurs thématiques

## 2. Visualisation des prévisions
- Graphique interactif comparant prévisions vs réalité
- Calcul des métriques MAE et RMSE
- Gestion du cas sans données réelles
