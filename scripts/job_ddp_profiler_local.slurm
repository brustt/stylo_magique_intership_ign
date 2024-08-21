#!/bin/bash
#SBATCH --mem=100G           # Memory (0=infinite=no other job possible in this node)
#SBATCH --partition=jean-zellou
#SBATCH --job-name=train_bisam_2 # nom du job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3       # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:3               # nbr of GPU per node
#SBATCH --cpus-per-task=16         # num_worker

#SBATCH --output=/mnt/common/hdd/home/SKhelifi/slurm_logs/slurm_logs_jean_zellou_output_train_bisam_%j.out # nom du fichier de sortie
#SBATCH --error=/mnt/common/hdd/home/SKhelifi/slurm_logs/slurm_logs_jean_zelloue_rror_train_bisam_cd_%j.out # nom du fichier d'erreur
#SBATCH --time=2:00:00 # temps maximum d'execution demande (HH:MM:SS)

# nettoyage des modules charges en interactif et herites par defaut
# Compulsory
# NOTE: Beginning with 22.05, srun will not inherit the --cpus-per-task value requested by salloc or sbatch.
# It must be requested again with the call to srun or set with the SRUN_CPUS_PER_TASK environment variable if desired for the task(s).
export SRUN_CPUS_PER_TASK=16

# For debug purposes (optional)
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=TRACE
# export SBATCH_DEBUG=1
# export PYTHONFAULTHANDLER=1

# Distributed error handlers (optional)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Network name
export NCCL_SOCKET_IFNAME=bond0



set -x 

export PROJECT_PATH=$HOME_SLURM/stage_stylo_magique_2024
export DATA_PATH=/mnt/stores/store-DAI/datasrc/dchan
export CHECKPOINTS_PATH=$HOME_SLURM/checkpoints/projects/stage_stylo_magique_2024
export LOGS_PATH=$HOME_SLURM/logs/projects/stage_stylo_magique_2024
export LOGS_JOB=$HOME_SLURM/job_logs/projects/stage_stylo_magique_2024
export PYTHONPATH=$PROJECT_PATH # could use https://github.com/ashleve/rootutils instead
export SAM_DATA_DEMO_PATH=""

export HYDRA_FULL_ERROR=1

mkdir -p ${CHECKPOINTS_PATH}
mkdir -p ${LOGS_PATH}
mkdir -p ${LOGS_JOB}

eval "$(conda shell.bash hook)"
conda activate mp
srun -vv python3 -c"import torch; print(torch.cuda.is_available());"