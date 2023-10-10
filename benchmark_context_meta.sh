#!/bin/bash

horizon=2500
nfolds=20




for context_type in  'MNISTbinary' #'linear' 'quadratic' 'sinusoid'

    do

    for game in  'LE' #'LE' 

        do

            for approach in  'random' #'cbpside' 'randcbpside' 'neurallincbpside' 'randneurallincbpside'
                    
                do

                python3 ./create_storage.py --horizon $horizon --n_folds $nfolds --game $game --approach $approach --task $task --context_type $context_type 
  
                for ((id=0; id<$nfolds; id+=4)) 

                    do

		            echo 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'GAME' $game 'TASK' $task 'APR' $approach 'ID' $id
    
                    sbatch --export=ALL,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,GAME=$game,TASK=$task,APR=$approach,ID=$id ./benchmark_context.sh     
                    
                    done

                done
                
            done
        
        done

    done
