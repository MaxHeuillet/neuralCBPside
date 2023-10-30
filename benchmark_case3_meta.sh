#!/bin/bash

horizon=9999
nfolds=1


for context_type in  'MNISTbinary' #'linear' 'quadratic' 'sinusoid'

    do

    for game in  'LE' 

        do

            for approach in 'random' #'random2' 'ineural' #  'ineural' 'margin' 'cesa'
                    
                do

                python3 ./create_storage.py  --case 'case3' --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type 
  
		        echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 

                #sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach ./benchmark_case3.sh     
                python3 ./benchmark_case3.py --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type

                done
                
        done
        
    done
