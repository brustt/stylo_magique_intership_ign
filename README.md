# stage_stylo_magique_2024

Repo du Stage "Stylo Magique" SIMV.

Equipe : 
* Martin Dizier
* Samy Khelifi
* Nicolas Gonthier

### Installation [Deprecated]

* Créer le dossier `models/`.
* Créer lien symbolique vers `/var/data/dl` dans `~/data`. Mettre à jour `root_data_path` au besoin (cf  `magic_pen/config.py`)
* Créer dossier de logs `lightnings_logs` à la racine du projet
* Télécharger les checkpoints SAM dans `models/` (voir ci dessous)
* Créer environenment conda depuis le fichier d'environnement : 
`mamba env create -f environment.yml`
* Activer environement `mp`
* Créer `.env` à la racine du projet
* Install packages du repo : `pip install -e .`


Checkpoints SAM : 
* default or vit_h: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
* vit_b: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

### Run
En attente...


### Set environment variable
create `.env` file and put project paths :

```
PROJECT_PATH = <..>
DATA_PATH = <..>
CHECKPOINTS_PATH = <..>
```

python = 3.10
