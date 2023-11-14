#!/bin/bash

horizon=9999
nfolds=25
case='case1'  # You can change this to 'case1', 'case2', 'case3', etc.


if [ "$case" == "case1" ]; then
    context_type='MNISTbinary'
else
    context_type='MNIST'
fi



for game in 'LE' 
    do
    for approach in 'EEneuralcbpside_v3' # 'ineural' 'EEneuralcbpside_v2' 'EEneuralcbpside_v4' 
        do
        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type 


        for ((id=0; id<$nfolds; id+=1)) 
            do
                echo 'case' $case 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'ID' $id
                sbatch --export=ALL,CASE=$case,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done

