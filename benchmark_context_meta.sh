#!/bin/bash

horizon=10000
nfolds=96
var=1

for context_type in 'linear' 'quadratic' 'sinusoid'

    do

    for game in  'AT' #'LE' 

        do 

            for task in 'balanced' 'imbalanced'

                do

                for alg in 'CBPside005' 'NeuralCBPsidev1' 'NeuralCBPsidev2' 'NeuralCBPsidev3'
                    
                    do
		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'ALG' $alg 'VAR' $var
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,ALG=$alg,VAR=$var ./partial_monitoring/benchmark_context.sh 
                    ((var++))
                    done
                
                done
        
        done

    done
