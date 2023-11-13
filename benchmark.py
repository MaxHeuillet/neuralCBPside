
import numpy as np
from multiprocess import Pool
# import multiprocessing as mp
import os

from functools import partial
import pickle as pkl
import gzip
import argparse
import os
import torch
import random

import games
import synthetic_data


# import cbpside
# import randcbpside2
# import cbpside
# import rand_cbpside
# import randneuralcbp
# import neuralcbp_LE
# import margin_based
# import rand_neural_lin_cbpside_disjoint
# import cesa_bianchi
import neuralcbp_EE_kclasses_v2
import neuralcbp_EE_kclasses_v4

import ineural_multi
import random_algo
import random_algo2


######################
######################


def evaluate_parallel(evaluator, game, nfolds, id):
    
    print('numbers of processes to be launched', nfolds)
    pool = Pool(processes=1)

    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    context_generators = []
    seeds = []
    algos = []

    gpu_id = 0

    for alg_id, seed in enumerate(range(id, id+1,1)):
        print(alg_id, seed)
        
        if evaluator.context_type == 'linear':
            size = 5
            w = np.array([1/size]*size)
            contexts = synthetic_data.LinearContexts( w , evaluator.task) 
            context_generators.append( contexts )

        elif evaluator.context_type == 'quadratic':
            size = 5
            w = np.array([1/size]*size)
            contexts = synthetic_data.QuadraticContexts( w , evaluator.task )
            context_generators.append( contexts )

        elif evaluator.context_type == 'sinusoid':
            size = 5
            w = np.array([1/size]*size)
            contexts = synthetic_data.SinusoidContexts( w , evaluator.task )
            context_generators.append( contexts )

        elif evaluator.context_type == 'MNISTbinary': 
            contexts = synthetic_data.MNISTcontexts_binary()
            context_generators.append( contexts )
            
        elif evaluator.context_type == 'MNIST': 
            contexts = synthetic_data.MNISTcontexts()
            context_generators.append( contexts )
        else:
            print('error')

        if args.approach == 'EEneuralcbpside_v2':
            lbd_neural = 0
            m = 100
            H = 50
            lbd_reg = 1
            nclasses = 10
            alg = neuralcbp_EE_kclasses_v2.CBPside( game, 1.01, lbd_neural, lbd_reg, m, H, nclasses,  'cuda:0')
            algos.append( alg )

        if args.approach == 'EEneuralcbpside_v4':
            lbd_neural = 0
            m = 100
            H = 50
            lbd_reg = 1
            nclasses = 10
            alg = neuralcbp_EE_kclasses_v4.CBPside( game, 1.01, lbd_neural, lbd_reg, m, H, nclasses,  'cuda:0')
            algos.append( alg )

        elif args.approach == 'ineural':
            budget = evaluator.horizon
            nclasses = 10
            alg = ineural_multi.INeurALmulti(budget, nclasses, 'cuda:0')
            algos.append( alg )

        seeds.append(seed)

    print('send jobs')
    print('seeds', context_generators, seeds, algos)
        
    pool.map( partial( evaluator.eval_policy_once, game ), zip(context_generators, seeds, algos ) ) 

    return True

class Evaluation:

    def __init__(self, case, game_name, n_folds, horizon, game, label, context_type):

        self.game_name = game_name
        self.n_folds = n_folds
        self.case = case
        self.horizon = horizon
        self.game = game
        self.label =  label
        self.context_type = context_type

    def get_outcomes(self, game, ):
        outcomes = np.random.choice( game.n_outcomes , p= list( game.outcome_dist.values() ), size= self.horizon) 
        return outcomes

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, game, job):

        print('start 1')
        context_generator, jobid, alg = job

        #print('start 2', alg.device)
        np.random.seed(jobid)
        torch.manual_seed(jobid)
        random.seed(jobid)

        alg.reset( context_generator.d )

        cumRegret =  np.zeros(self.horizon, dtype =float)
        print('start 3')

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context, distribution = context_generator.get_context()
            outcome = np.argmax(distribution) 

            context = np.expand_dims(context, axis=0)
            #print('context shape', context.shape)
            
            action, _ = alg.get_action(t, context)

            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )

            print('t', t, 'action', action, 'outcome', outcome, 'gaps', ( game.LossMatrix[0,...] - game.LossMatrix[1,...])  @ distribution  )

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            cumRegret[t] =  val

        result = np.cumsum(cumRegret)
        print(result)
        print('finished', jobid)
        with gzip.open( './results/{}_{}_{}_{}_{}.pkl.gz'.format(self.case,self.context_type, self.horizon, self.n_folds, self.label) ,'ab') as f:
            pkl.dump(result,f)
        print('saved', jobid)

        return True


###################################
# Synthetic Contextual experiments
###################################

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

parser = argparse.ArgumentParser()

parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--game", required=True, help="game")
parser.add_argument("--case", required=True, help="case")
parser.add_argument("--context_type", required=True, help="context type")
parser.add_argument("--approach", required=True, help="algorithme")
parser.add_argument("--id", required=True, help="algorithme")

args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)
id = int(args.id)
print(args.context_type, args.approach)


if args.case == 'case2':
    game = games.game_case2(  )
elif args.case == 'case3':
    game = games.game_case3(  )
elif args.case == 'case4':
    game = games.game_case4(  )



# factor_type = args.approach.split('_')[1]
# print('factor_type', factor_type)

ncpus = int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
ngpus = int( torch.cuda.device_count() )
# nfolds = 5 #min([ncpus,ngpus]) 
print('ncpus', ncpus,'ngpus', ngpus)


evaluator = Evaluation(args.case, args.game, n_folds, horizon, game, args.approach, args.context_type)

evaluate_parallel(evaluator, game, n_folds, id)
        
