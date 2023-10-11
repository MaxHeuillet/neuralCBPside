#!/bin/bash

#SBATCH --account=def-adurand

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --gpus-per-node=0

#SBATCH --mem-per-cpu=3072M
#SBATCH --time=01:00:00

#SBATCH --mail-user=maxime.heuillet.1@ulaval.ca
#SBATCH --mail-type=ALL


echo 'horizon' ${HORIZON} 'nfolds' ${NFOLDS} 'CONTEXT_TYPE' ${CONTEXT_TYPE} 'GAME' ${GAME} 'TASK' ${TASK} 'APPROACH' ${APR} 


module --force purge
module load StdEnv/2020
module load python/3.10
module load scipy-stack
module load gurobi
source /home/mheuill/projects/def-adurand/mheuill/MYENV2/bin/activate

virtualenv-clone /home/mheuill/projects/def-adurand/mheuill/MYENV2 $SLURM_TMPDIR/MYENV2
deactivate
source $SLURM_TMPDIR/MYENV2/bin/activate

echo 'HZ: start python3 ./experiment.py ..at '; date

python3 ./benchmark_context.py --horizon ${HORIZON} --n_folds ${NFOLDS} --game ${GAME} --approach ${APR} --task ${TASK} --context_type ${CONTEXT_TYPE} --id ${ID} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID

