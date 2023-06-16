
import numpy as np


from multiprocess import Pool
#import multiprocessing as mp
import os

from functools import partial
import pickle as pkl
import gzip

import games

import cbpside
# import randcbpside2

import synthetic_data

import gzip
import pickle as pkl

import subprocess


import neuralcbpside_v1
import neuralcbpside_v2
import neuralcbpside_v3


import argparse
import os
import torch

######################
######################


def reshape_context(context, A):
    d = context.shape[0]
    reshaped_context = np.zeros((A, d * A))
    
    for i in range(A):
        start_idx = i * d
        end_idx = start_idx + d
        reshaped_context[i, start_idx:end_idx] = context
    
    return reshaped_context

def evaluate_parallel(evaluator, game, nfolds, id):

    
    pool = Pool(processes=nfolds)

    np.random.seed(1)
    context_generators = []
    alg_ids =[]
    seeds = []
    algos = []
    size = 5
    w = np.array([1/size]*size)

    for alg_id, seed in enumerate(range(id, id+nfolds,1)):
        
        if evaluator.context_type == 'linear':
            contexts = synthetic_data.LinearContexts( w , evaluator.task) 
            context_generators.append( contexts )

        elif evaluator.context_type == 'quadratic':
            contexts = synthetic_data.QuadraticContexts( w , evaluator.task )
            context_generators.append( contexts )

        else: 
            contexts = synthetic_data.SinusoidContexts( w , evaluator.task )
            context_generators.append( contexts )

        if 'neural' in args.approach:
            algos.append( neuralcbpside_v3.NeuralCBPside(game, factor_type, 1.01, 0.05, 5, "cuda:{}".format(alg_id) ) )
        else:
            algos.append( cbpside.CBPside(game, 1.01, 0.05 )  )

        seeds.append(seed)
        alg_ids.append(alg_id)

    print('send jobs')
        
    return pool.map( partial( evaluator.eval_policy_once, game ), zip(context_generators, seeds, algos ) ) 

class Evaluation:

    def __init__(self, game_name, task, n_folds, horizon, game, label, context_type):

        self.game_name = game_name
        self.task = task
        self.n_folds = n_folds
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

        # print('start 2', alg.device)
        np.random.seed(jobid)

        alg.reset( context_generator.d )

        cumRegret =  np.zeros(self.horizon, dtype =float)
        print('start 3')

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context, distribution = context_generator.get_context()

            outcome = 0 if distribution[0]>0.5 else 1  #np.random.choice( 2 , p = distribution )
            
            context = reshape_context(context, alg.A) if 'neural' in alg.name else np.reshape(context, (-1,1))

            action = alg.get_action(t, context)

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
        with gzip.open( './results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(self.game_name, self.task, self.context_type, self.horizon, self.n_folds, self.label) ,'ab') as f:
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
parser.add_argument("--task", required=True, help="task")
parser.add_argument("--context_type", required=True, help="context type")
parser.add_argument("--approach", required=True, help="algorithme")
parser.add_argument("--id", required=True, help="algorithme")

args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)
id = int(args.id)
print(id, args.context_type, args.approach)

games = {'AT':games.apple_tasting()} #'LE': games.label_efficient(  ),
game = games[args.game]

factor_type = args.approach.split('_')[1]
print('factor_type', factor_type)


ncpus = os.environ.get('SLURM_CPUS_PER_TASK',default=1)
ngpus = torch.cuda.device_count()
if 'neural' in args.approach:
    nfolds = min([ncpus,ngpus]) 
else:
    nfolds = ncpus

print('nfolds', nfolds)

evaluator = Evaluation(args.game, args.task, n_folds, horizon, game, args.approach, args.context_type)

# with gzip.open( './results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach) ,'wb') as g:
#     pkl.dump( [None]*horizon, g)

evaluate_parallel(evaluator, game, nfolds, id)
        
