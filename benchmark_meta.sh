#!/bin/bash


horizon=9999
nfolds=$1


#######################################
##### Cost-sensitive experiment
#######################################


# case='case1b'
# model='MLP'

# context_types=('adult' 'MagicTelescope' 'MNISTbinary' )
# approaches=( 'EEneuralcbpside_v6' 'ineural3' 'ineural6' 'margin' 'cesa' 'neuronal3' 'neuronal6' )  


# for context_type in "${context_types[@]}"; do

#     for approach in "${approaches[@]}"; do

#         python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 

#         for ((id=0; id<$nfolds; id+=1)); do
#             echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
#             sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
#             done

#     done
# done

#######################################
##### To evaluate on the binary tasks with MLP
#######################################


case='case1'
model='MLP'


context_types=('adult' 'MagicTelescope' 'MNISTbinary' ) # 
approaches=( 'EEneuralcbpside_v6' )  # 'neuronal6'  'ineural3' 'ineural6' 'cesa'  'margin'   'neuronal3'


for context_type in "${context_types[@]}"; do

    for approach in "${approaches[@]}"; do

        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 

        for ((id=0; id<$nfolds; id+=1)); do
            echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
            sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done

#######################################
##### To evaluate on the 10-classes tasks with MLP
#######################################

case='case2'
model='MLP'

context_types=( 'MNIST' 'FASHION' ) 
approaches=(  'EEneuralcbpside_v6'  )  #'ineural3' 'ineural6' 'neuronal3' 'neuronal6'

for context_type in "${context_types[@]}"; do

    for approach in "${approaches[@]}"; do

        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 

        for ((id=0; id<$nfolds; id+=1)); do
            echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
            sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
    
done

#######################################
##### To evaluate on the other tasks with MLP
#######################################

case='game_case_seven'
model='MLP'

context_types=('shuttle' 'covertype' )
approaches=( 'EEneuralcbpside_v6' ) #'ineural3' 'ineural6' 'neuronal3' 'neuronal6'

for context_type in "${context_types[@]}"; do

    for approach in "${approaches[@]}"; do

        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 

        for ((id=0; id<$nfolds; id+=1)); do
            echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
            sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done

#######################################
##### To evaluate with LeNet on MNIST, Fashion-MNIST, CIFAR-10 
#######################################

horizon=9999
nfolds=$1

case='case2'
model='LeNet'


context_types=('CIFAR10' 'MNIST' 'FASHION') 
approaches=('EEneuralcbpside_v6' )  #'ineural3' 'ineural6' 'neuronal3' 'neuronal6'

for context_type in "${context_types[@]}"; do

    for approach in "${approaches[@]}"; do

        python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 

        for ((id=0; id<$nfolds; id+=1)); do
            echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
            sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
            done

    done
done


#######################################
##### To evaluate on all of them:
#######################################

# horizon=9999
# nfolds=$1
# context_type=$2
# case=$3
# model=$4

# if [ "$model" == "MLP" ]; then
#     if [ "$case" == "case1" ]; then
#         approaches=('ineural3' 'ineural6' ) #'margin' 'cesa'  'neuronal3' 'neuronal6' 'EEneuralcbpside_v6'
#     else
#         approaches=('ineural3' 'ineural6' ) #'neuronal3' 'neuronal6' 'EEneuralcbpside_v6'
#     fi
# else
#     if [ "$case" == "case1" ]; then
#         approaches=('neuronal3' 'neuronal6' 'EEneuralcbpside_v6')
#     else
#         approaches=('neuronal3' 'neuronal6' 'EEneuralcbpside_v6')
#     fi
# fi


# for approach in "${approaches[@]}"; do
#     python3 ./create_storage.py --case $case --horizon $horizon --n_folds $nfolds --model $model --approach $approach --context_type $context_type 


#     for ((id=0; id<$nfolds; id+=1)); do
#         echo 'case' $case 'model' $model 'horizon' $horizon 'nfolds' $nfolds 'CONTEXT_TYPE' $context_type 'TASK' $task 'APR' $approach 'ID' $id
#         sbatch --export=ALL,CASE=$case,MODEL=$model,HORIZON=$horizon,NFOLDS=$nfolds,CONTEXT_TYPE=$context_type,APR=$approach,ID=$id ./benchmark_launch.sh     
#         done

# done






