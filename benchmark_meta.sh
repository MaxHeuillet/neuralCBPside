#!/bin/bash

horizon=9999
nfolds=$1
context_type=$2
case=$3
model=$4

if [ "$case" == "case1" ]; then
    approaches=( 'neuronal3'  'neuronal6' 'EEneuralcbpside_v6') #'margin' 'cesa' 'ineural3' 'ineural6'  'EEneuralcbpside_v6' 'EEneuralcbpside_v5' 'ineural3' 'ineural6' 'neuronal3' 'neuronal6' 
else   
    approaches=(  'neuronal3'  'neuronal6' 'EEneuralcbpside_v6') #'ineural3' 'ineural6' 'EEneuralcbpside_v6' 'EEneuralcbpside_v5'  'ineural3' 'ineural6' 'neuronal3' 'neuronal6' 
fi

# for model in 'MLP' 'LeNet' ; do 
#     for approach in "${approaches[@]}"; do
#         python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


#         for ((id=0; id<$nfolds; id+=1)); do
#                 echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
#                 sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
#             done

#     done
# done




for approach in "${approaches[@]}"; do
    python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


    for ((id=0; id<$nfolds; id+=1)); do
        echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
        sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
        done

done






