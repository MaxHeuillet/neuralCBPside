#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=300M

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL


echo 'horizon' ${HORIZON} 'nfolds' ${NFOLDS} 'CONTEXT_TYPE' ${CONTEXT_TYPE} 'GAME' ${GAME} 'TASK' ${TASK} 'ALG' ${ALG} 

module --force purge

module load StdEnv/2020

module load python/3.10

module load scipy-stack

module load gurobi

source /home/mheuill/projects/def-adurand/mheuill/ENV_nogurobi/bin/activate

python3 ./benchmark_context.py --horizon ${HORIZON} --n_folds ${NFOLDS} --game ${GAME} --alg ${ALG} --task ${TASK} --context_type ${CONTEXT_TYPE} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID
