#!/bin/bash

horizon=2500
nfolds= 8
nfolds_per_node=$((nfolds / 4)) # each node has 4 GPUs




for context_type in 'linear' #'quintic'

    do

    for game in  'AT' #'LE' 

        do 

        for task in 'imbalanced' #'balanced'

            do

            for approach in 'neuralcbp_simplified' 'neuralcbp_1' #'neuralcbp_theory'
                    
                do

                python3 ./create_storage.py --horizon $horizon --n_folds $nfolds --game $game --approach $approch --task $task --context_type $context_type 
  
                # for ((i=0; i<n_folds_per_node; i++))

                #     do

		        #     echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach
    
                #     sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach ./benchmark_context.sh     
                    
                #     done

                done
                
            done
        
        done

    done
