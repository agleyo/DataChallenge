# Structure du projet
`train.csv` : Données de consommation électrique utilisées pour entrainer le modèle.

`test.csv` : Données utilisées pour évaluer la performance du modèle.

`data.csv` : Données sur les périodes de vacances scolaires.

`meteo.parquet` : Données météorologiques.

`Regions.csv` et `Departements.csv` : Données démographiques portant sur les régions et départements en 2017 selon l’INSEE.

`predict.py` : Script python permettant de générer les prédictions à partir des modèles entrainés.

`train.py` : Script python permettant d’entrainer les modèles.

`data.py` : Script python générant le jeu de données.

`interpolatedData/` : Dossier contenant les données météorologiques et de consommation électrique pour chaque région.

`models/` : Dossier contenant les modèles entrainés pour chaque région.

`solution_template/` : Dossier contenant le fichier de prédiction et les checkpoints.


# Execution du script et prédiction

Utiliser le fichier `predict.py`.

# Auteurs

Le projet a été développé par Alexis Gleyo et Simon Legris.
