#!/bin/bash

#SBATCH --account=def-adurand
#SBATCH --gpus-per-node=4
#SBATCH --mem=400M
#SBATCH --time=00:25:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
module load gurobi
source /home/mheuill/projects/def-adurand/mheuill/ENV_nogurobi/bin/activate

virtualenv-clone /home/mheuill/projects/def-adurand/mheuill/ENV_nogurobi $SLURM_TMPDIR/ENV_nogurobi
deactivate
source $SLURM_TMPDIR/ENV_nogurobi/bin/activate

echo 'HZ: start python3 ./experiment.py ..at '; date

python3 ./benchmark_context.py --horizon ${HORIZON} --n_folds ${NFOLDS} --game ${GAME} --approach ${APR} --task ${TASK} --context_type ${CONTEXT_TYPE} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

