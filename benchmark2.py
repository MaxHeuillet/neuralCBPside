
import numpy as np
import os
import argparse
import os
import torch

import games
import synthetic_data
import evaluator

import ineural_multi
import neuronal
import margin_based
import cesa_bianchi
import neuralcbp_EE_kclasses_v2
import neuralcbp_EE_kclasses_v3
import neuralcbp_EE_kclasses_v4
import neuralcbp_EE_kclasses_v5
import neuralcbp_EE_kclasses_v6

# import cbpside
# import randcbpside2
# import cbpside
# import rand_cbpside
# import randneuralcbp
# import neuralcbp_LE
# import rand_neural_lin_cbpside_disjoint
# import random_algo
# import random_algo2


###################################
# Synthetic Contextual experiments
###################################

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--model", required=True, help="model")
parser.add_argument("--case", required=True, help="case")
parser.add_argument("--context_type", required=True, help="context type")
parser.add_argument("--approach", required=True, help="algorithme")
parser.add_argument("--id", required=True, help="algorithme")

args = parser.parse_args()

ncpus = int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
ngpus = int( torch.cuda.device_count() )
print('ncpus', ncpus,'ngpus', ngpus)

############################# INITIATE THE EXPERIMENT:

print('nfolds', args.n_folds)
horizon = int(args.horizon)
n_folds = int(args.n_folds)
seed = int(args.id)
print(args.context_type, args.approach)

if args.case == 'case1':
    game = games.game_case1( {} )
elif args.case == 'case2':
    game = games.game_case2( {} )
elif args.case == 'case3':
    game = games.game_case3( {} )
elif args.case == 'case4':
    game = games.game_case4( {} )
elif args.case == 'game_case_seven':
    game = games.game_case_seven( {} )

eval = evaluator.Evaluation(args.case, args.model, n_folds, horizon, game, args.approach, args.context_type)
eval.set_random_seeds(seed)

################################### CONTEXT GENERATOR:


if args.context_type == 'MNISTbinary':
    context_generator = synthetic_data.MNISTcontexts_binary(eval)
    context_generator.initiate_loader()


elif args.context_type == 'adult':
    data = synthetic_data.Bandit_multi('adult') 
    context_generator = synthetic_data.CustomContexts_binary(eval)
    context_generator.initiate_loader(data.X, data.y)
    

elif args.context_type == 'MagicTelescope':
    data = synthetic_data.Bandit_multi('MagicTelescope') 
    context_generator = synthetic_data.CustomContexts_binary(eval)
    context_generator.initiate_loader(data.X, data.y)

elif args.context_type == 'covertype':
    data = synthetic_data.Bandit_multi('covertype') 
    context_generator = synthetic_data.CustomContexts_binary(eval)
    context_generator.initiate_loader(data.X, data.y)

elif args.context_type == 'shuttle':
    data = synthetic_data.Bandit_multi('shuttle') 
    context_generator = synthetic_data.CustomContexts_binary(eval)
    context_generator.initiate_loader(data.X, data.y)


elif args.context_type == 'MNIST':
    context_generator = synthetic_data.MNISTcontexts(eval)
    context_generator.initiate_loader()

elif args.context_type == 'FASHION':
    context_generator = synthetic_data.FashionMNISTContexts(eval)
    context_generator.initiate_loader()

elif args.context_type == 'CIFAR10':
    context_generator = synthetic_data.CIFAR10Contexts(eval)
    context_generator.initiate_loader()


################################### AGENT:

m = 100
nclasses = game.M

# if args.approach == 'EEneuralcbpside_v2':
#     alg = neuralcbp_EE_kclasses_v2.CBPside( game, 1.01, m, nclasses,  'cuda:0')


# elif args.approach == 'EEneuralcbpside_v3':
#     alg = neuralcbp_EE_kclasses_v3.CBPside( game, 1.01, m, nclasses,  'cuda:0')


# elif args.approach == 'EEneuralcbpside_v4':
#     alg = neuralcbp_EE_kclasses_v4.CBPside( game, 1.01, m, nclasses,  'cuda:0')


# elif args.approach == 'EEneuralcbpside_v5':
#     alg = neuralcbp_EE_kclasses_v5.CBPside( game, 1.01, m, nclasses,  'cuda:0')


if args.approach == 'EEneuralcbpside_v6':
    alg = neuralcbp_EE_kclasses_v6.CBPside( game, args.context_type, eval.model, 1.01, m, nclasses,  'cuda:0')

elif args.approach == 'ineural3':
    budget = eval.horizon
    margin = 3
    alg = ineural_multi.INeurALmulti(budget, nclasses, margin, m, 'cuda:0')

elif args.approach == 'ineural6':
    budget = eval.horizon
    margin = 6
    alg = ineural_multi.INeurALmulti(budget, nclasses, margin, m, 'cuda:0')


elif args.approach == 'neuronal3':
    budget = eval.horizon
    margin = 3
    alg = neuronal.NeuronAL(eval.model,args.context_type, budget, nclasses, margin, m,'cuda:0')

elif args.approach == 'neuronal6':
    budget = eval.horizon
    margin = 6
    alg = neuronal.NeuronAL(eval.model,args.context_type, budget, nclasses, margin, m,'cuda:0')

elif args.approach == 'margin':
    threshold = 0.1
    alg = margin_based.MarginBased(game, m, threshold,  'cuda:0')

elif args.approach == 'cesa':
    alg = cesa_bianchi.CesaBianchi(game, m, 'cuda:0')



alg.reset(context_generator.d)


job = context_generator, alg 
eval.eval_policy_once( game, job )  
        
