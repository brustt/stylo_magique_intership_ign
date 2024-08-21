#!/bin/bash
#SBATCH --account=ofp@v100
#SBATCH --job-name=train_bisam_2 # nom du job
##SBATCH -C v100-32g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4               # nbr of GPU per node
#SBATCH --cpus-per-task=16         # num_worker
#SBATCH --hint=nomultithread       # physical core
#SBATCH --output=/lustre/fsn1/projects/rech/ofp/commun/job_logs/samy/stage_stylo_magique_2024/output_train_bisam_cd_%j.out # nom du fichier de sortie
#SBATCH --error=/lustre/fsn1/projects/rech/ofp/commun/job_logs/samy/stage_stylo_magique_2024/error_train_bisam_cd_%j.out # nom du fichier d'erreur
#SBATCH --time=2:00:00 # temps maximum d'execution demande (HH:MM:SS)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement de l'environnement
module load pytorch-gpu/py3/2.0.0

set -x 

# SET MAIN PATHS
export PROJECT_PATH=$NEWSCRATCH/projects/stage_stylo_magique_2024
export DATA_PATH=$ALL_CCFRNEWSCRATCH/datasets
export CHECKPOINTS_PATH=$ALL_CCFRNEWSCRATCH/checkpoints/samy/stage_stylo_magique
export LOGS_PATH=$ALL_CCFRNEWSCRATCH/logs/samy/stage_stylo_magique_2024
export LOGS_JOB=
export PYTHONPATH=$PROJECT_PATH # could use https://github.com/ashleve/rootutils instead
export SAM_DATA_DEMO_PATH=""

# SET HYDRA ERROR LEVEL
export HYDRA_FULL_ERROR=1

# NCCL optimization, see https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
# END NCCL optimization

# CREATE IF NOT EXISTS MAIN PATHS (with parents if necessary)
mkdir -p ${CHECKPOINTS_PATH}
mkdir -p ${LOGS_PATH}
mkdir -p ${LOGS_JOB}

# EXECUTE SCRIPTS
srun time python $PROJECT_PATH/src/train.py experiment=probing_diff sam_type=small data=levir-cd trainer=ddp task_name=freeze_test trainer.max_epochs=1
