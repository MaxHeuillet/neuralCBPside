#!/bin/bash

horizon=9999
nfolds=12


for context_type in  'MNIST' #'linear' 'quadratic' 'sinusoid'

    do

    for game in  'LE' 

        do

            for approach in 'EEneuralcbpside'
                    
                do

                python3 ./create_storage.py  --case 'case3' --horizon $horizon --n_folds $nfolds --game $game --approach $approach --context_type $context_type 
  
		        for ((id=0; id<$nfolds; id+=1)) 

                    do

		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'ID' $id

                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,APR=$approach,ID=$id ./benchmark_case32.sh     

                    done

                done
                
        done
        
    done

