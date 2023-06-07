#!/bin/bash
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
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


# module --force purge

# module load nixpkgs/16.09

# module load python/3.7

# module load scipy-stack

# module load gurobi/9.0.1

# source /home/mheuill/projects/def-adurand/mheuill/ENV_gurobi/bin/activate



# module load nixpkgs/16.09
#gcc/7.3.0 ipopt/3.12.13
#virtualenv --no-download ./env${VAR}
#source ./env${VAR}/bin/activate
#pip install --no-index scikit-learn
#cd $EBROOTGUROBI
#python setup.py build --build-base /tmp/${USER} install

# echo "Threads 1" > gurobi.env   # set number of threads


#echo "$PWD"
#cd ~/projects/def-adurand/mheuill/attack-detection
#echo "$PWD"

python3 ./partial_monitoring/benchmark_context.py --horizon ${HORIZON} --n_folds ${NFOLDS} --game ${GAME} --alg ${ALG} --task ${TASK} --context_type ${CONTEXT_TYPE} > stdout_$SLURM_JOB_ID 2>stderr_$SLURM_JOB_ID
