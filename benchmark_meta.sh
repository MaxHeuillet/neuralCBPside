#!/bin/bash

#######################################
##### To evaluate on the binary tasks
#######################################

horizon=9999
nfolds=$1

case='case1'
model='MLP'


context_types=('adult' 'MagicTelescope' 'MNISTbinary' )
approach=( 'margin' 'cesa' 'ineural3' 'ineural6' 'neuronal3' 'neuronal6' 'EEneuralcbpside_v6')


for context_type in "${context_types[@]}"; do

    for approach in "${approaches[@]}"; do

        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


        for ((id=0; id<$nfolds; id+=1)); do
            echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
            sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done


#######################################
##### To evaluate on all of them:
#######################################

# horizon=9999
# nfolds=$1
# context_type=$2
# case=$3
# model=$4

# if [ "$model" == "MLP" ]; then
#     if [ "$case" == "case1" ]; then
#         approaches=('ineural3' 'ineural6' ) #'margin' 'cesa'  'neuronal3' 'neuronal6' 'EEneuralcbpside_v6'
#     else
#         approaches=('ineural3' 'ineural6' ) #'neuronal3' 'neuronal6' 'EEneuralcbpside_v6'
#     fi
# else
#     if [ "$case" == "case1" ]; then
#         approaches=('neuronal3' 'neuronal6' 'EEneuralcbpside_v6')
#     else
#         approaches=('neuronal3' 'neuronal6' 'EEneuralcbpside_v6')
#     fi
# fi


# for approach in "${approaches[@]}"; do
#     python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


#     for ((id=0; id<$nfolds; id+=1)); do
#         echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
#         sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
#         done

# done






