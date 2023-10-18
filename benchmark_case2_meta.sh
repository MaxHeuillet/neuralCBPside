#!/bin/bash

horizon=9999
nfolds=10


for context_type in  'MNIST' #'linear' 'quadratic' 'sinusoid'

    do

    for game in  'LE' 

        do

            for approach in 'randneuralcbpside' 'ineural' 'random' #'neuralcbpside'
                    
                do

                python3 ./create_storage.py  --case 'case2' --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type 
  
		        echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 

                sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach ./benchmark_case2.sh     

                done
                
        done
        
    done

