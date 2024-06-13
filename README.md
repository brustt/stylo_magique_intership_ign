# stage_stylo_magique_2024

Repo du Stage "Stylo Magique" SIMV.

Equipe : 
* Martin Dizier
* Samy Khelifi
* Nicolas Gonthier

### Installation

* Créer les dossiers `models/`, `data/`.
* Télécharger les checkpoints SAM dans `models/`
* Créer environenment conda depuis le fichier d'environnement : 
`mamba env create -f environment.yml`
* activer environement mp
* créer `.env` à la racine du projet
* install packages du repo : `pip install -e .`

### Run
#### Mode main (inference + eval) - lightning

```bash
python src/segment_any_change/main.py --ds_name=levir-cd --batch_size=2 --n_job_by_node=1 --dev
```
```
Paramètres : 

* --ds_name : nom dataset - valid names : levir-cd | second
* --batch_size : taille batch - attention consommation mémoire
* --n_job_by_node : nombre de job par noeud de compute
* --dev (`argparse.BooleanOptionalAction`) : si spécifié, run expérience avec vit_b et une petit grille de prompt (25)
```
#### Mode inference sample debug (inférence sur sous-ensemble de données définis par indices)

Configurez paramètres de run depuis `inference.py`.

```bash
python src/segment_any_change/inference.py 
```

python = 3.10
