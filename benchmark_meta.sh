#!/bin/bash

horizon=9999
nfolds=1
case='case1'  # You can change this to 'case1', 'case2', 'case3', etc.


if [ "$case" == "case1" ]; then
    context_type='MNISTbinary'
    approaches=( 'neuronal3' ) #'EEneuralcbpside_v6' 'EEneuralcbpside_v5' 'ineural3' 'ineural6' 'neuronal3' 'neuronal6' 'margin' 'cesa'
else
    context_type='MNIST'
    approaches=( 'neuronal3' ) #'EEneuralcbpside_v6' 'EEneuralcbpside_v5'  'ineural3' 'ineural6' 'neuronal3' 'neuronal6' 
fi

for model in 'MLP' 'LeNet'; do
    for approach in "${approaches[@]}"; do
        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


        for ((id=0; id<$nfolds; id+=1)); do
                echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
                sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done

