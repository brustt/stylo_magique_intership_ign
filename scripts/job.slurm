#!/bin/bash 

#SBATCH --account=ofp@v100
#SBATCH --job-name=seganychange # nom du job 

##SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:4               # nbr of GPU per node
#SBATCH --cpus-per-task=8         # nbr of cores per task
#SBATCH --hint=nomultithread        # physical core

#SBATCH --output=$SCRATCH/output_%j.txt) # nom du fichier de sortie 
#SBATCH --error=$SCRATCH/error_%j.txt) # nom du fichier d'erreur 
#SBATCH --time=3:00:00 # temps maximum d'execution demande (HH:MM:SS) 

# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load python
 
# echo des commandes lancees
conda deactivate
conda activate magic_pen

set -x 

export PROJECT_PATH=$SCRATCH/projects/stage_stylo_magique_2024
export DATA_PATH=$SCRATCH/data
export CHECKPOINTS_PATH=$WORK/checkpoints
export LOGS_PATH=$WORK/logs
export LOGS_JOB=$WORK/job_logs

export HYDRA_FULL_ERROR=1

mkdir -p ${LOGS_PATH}
mkdir -p ${LOGS_JOB}



##SBATCH --partition=preproc
##SBATCH --nodes=4 # on demande un noeud 
##SBATCH --ntasks-per-node=1 # avec une tache par noeud (= nombre de GPU ici) 
##SBATCH --cpus-per-task=4 # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU) 



srun python $PROJECT_PATH/src/eval.py experiment=seganychange_prompt  sam_type=small data=levir-cd debug=seganychange_prompt
