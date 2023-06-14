#!/bin/bash

horizon=2500
nfolds=48
var=1

for context_type in 'linear' #'quintic'

    do

    for game in  'AT' #'LE' 

        do 

            for task in 'balanced' 'imbalanced'

                do

                for approach in 'neuralcbp_simplified' 'neuralcbp_1' #'neuralcbp_theory'
                    
                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'VAR' $var
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach,VAR=$var ./benchmark_context.sh 
                    ((var++))
                    done
                
                done
        
        done

    done
