#!/bin/bash

horizon=2500
nfolds=8




for context_type in 'linear' #'quintic'

    do

    for game in  'AT' #'LE' 

        do 

        for task in 'imbalanced' #'balanced'

            do

            for approach in 'neuralcbp_simplified' 'neuralcbp_1' #'neuralcbp_theory'
                    
                do

                python3 ./create_storage.py --horizon $horizon --n_folds $nfolds --game $game --approach $approach --task $task --context_type $context_type 
  
                for id in ((id=0; id<20; id+=4))

                    do

		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'ID' $id
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach,ID=$id ./benchmark_context.sh     
                    
                    done

                done
                
            done
        
        done

    done
