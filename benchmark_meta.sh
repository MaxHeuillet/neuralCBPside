#!/bin/bash

horizon=9999
nfolds=25
case='case4'  # You can change this to 'case1', 'case2', 'case3', etc.

for context_type in 'MNIST'  # Add other context types if needed
do
    for game in 'LE' 
    do
        for approach in 'ineural' 'EEneuralcbpside_v2' 'EEneuralcbpside_v4'  # Add other approaches if needed
        do
            python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type 

            # if [[ $approach == 'EEneuralcbpside_v2' || $approach == 'EEneuralcbpside_v4' ]]; then

            for ((id=0; id<$nfolds; id+=1)) 
                do
                    echo 'case' $case 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'ID' $id
                    sbatch --export=ALL,CASE=$case,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,APR=$approach,ID=$id ./benchmark_launch.sh     
                done

            # else
            #     echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 
            #     sbatch --export=ALL,CASE=$case,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach ./benchmark_other.sh     
            # fi
        done
    done
done

