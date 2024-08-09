#!/bin/bash 

#SBATCH --account=ofp@v100
#SBATCH --job-name=train_bisam # nom du job 

#SBATCH -C v100-32g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1        # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4               # nbr of GPU per node
#SBATCH --cpus-per-task=16          # num_worker
#SBATCH --hint=nomultithread       # physical core

#SBATCH --output=/gpfswork/rech/ofp/urz33hl/job_logs/output_train_bisam_%j.out # nom du fichier de sortie 
#SBATCH --error=/gpfswork/rech/ofp/urz33hl/job_logs/error_train_bisam_cd_%j.out # nom du fichier d'erreur 
#SBATCH --time=20:00:00 # temps maximum d'execution demande (HH:MM:SS) 

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement de l'environnement
module load pytorch-gpu/py3/2.0.0

set -x 

export PROJECT_PATH=$NEWSCRATCH/projects/stage_stylo_magique_2024
export DATA_PATH=$ALL_CCFRNEWSCRATCH/datasets
export CHECKPOINTS_PATH=$WORK/checkpoints
export LOGS_PATH=$WORK/logs
export LOGS_JOB=$WORK/job_logs
export PYTHONPATH=$PROJECT_PATH # could use https://github.com/ashleve/rootutils instead
export SAM_DATA_DEMO_PATH=""

export HYDRA_FULL_ERROR=1

mkdir -p ${LOGS_PATH}
mkdir -p ${LOGS_JOB}


srun python $PROJECT_PATH/src/train.py experiment=probing_diff sam_type=small data=levir-cd trainer=ddp
