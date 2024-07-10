#!/bin/bash 
export PROJECT_PATH=$SCRATCH/projects/stage_stylo_magique_2024
export DATA_PATH=$SCRATCH/data
export CHECKPOINTS_PATH=$WORK/checkpoints
export OUT_DIR=$WORK/outjob_logs

mkdir -p ${OUTPUT_DIR}

#SBATCH --account=ofp@v100
#SBATCH --job-name=seganychange # nom du job 
#SBATCH --partition=preproc

#SBATCH --output=$(echo ${OUTPUT_DIR}/output_%j.txt) # nom du fichier de sortie 
#SBATCH --error=$(echo ${OUTPUT_DIR}/error_%j.txt) # nom du fichier d'erreur 
#SBATCH --nodes=4 # on demande un noeud 

#SBATCH --ntasks-per-node=1 # avec une tache par noeud (= nombre de GPU ici) 
#SBATCH --cpus-per-task=4 # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU) 
#SBATCH --time=20:00:00 # temps maximum d'execution demande (HH:MM:SS) 



srun python $PROJECT_PATH/src/main.py