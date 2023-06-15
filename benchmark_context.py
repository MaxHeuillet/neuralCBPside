
import numpy as np


from multiprocess import Pool
#import multiprocessing as mp
import os

from functools import partial
import pickle as pkl
import gzip

import games

import cbpside
import randcbpside2

import synthetic_data

import gzip
import pickle as pkl

import subprocess


import neuralcbpside_v1
import neuralcbpside_v2
import neuralcbpside_v3


import argparse
import os

######################
######################

def evaluate_parallel(evaluator, algos, game, id):

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print('ncpus',ncpus)
    
    pool = Pool(processes=ncpus)

    np.random.seed(1)
    context_generators = []
    seeds = []
    size = 5
    w = np.array([1/size]*size)

    for seed in range(id, id+4,1):
        
        if evaluator.context_type == 'linear':
            contexts = synthetic_data.LinearContexts( w , evaluator.task) 
            context_generators.append( contexts )

        elif evaluator.context_type == 'quadratic':
            contexts = synthetic_data.QuadraticContexts( w , evaluator.task )
            context_generators.append( contexts )

        else: 
            contexts = synthetic_data.SinusoidContexts( w , evaluator.task )
            context_generators.append( contexts )

        seeds.append(seed)
        
    return  pool.map( partial( evaluator.eval_policy_once, game ), zip(algos, context_generators, seeds ) ) 

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

        alg, context_generator, jobid = job

        np.random.seed(jobid)

        alg.reset( context_generator.d )

        cumRegret =  np.zeros(self.horizon, dtype =float)

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context, distribution = context_generator.get_context()
            outcome = 0 if distribution[0]<0.5 else 1 #np.random.choice( 2 , p = distribution )

            action = alg.get_action(t, context)

            # print('t', t, 'action', action, 'outcome', outcome, )
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            cumRegret[t] =  val

        result = np.cumsum( cumRegret)
        with gzip.open( './results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(self.game_name, self.task, self.context_type, self.horizon, self.n_folds, self.label) ,'ab') as f:
            pkl.dump(result,f)

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
print(id)

games = {'AT':games.apple_tasting()} #'LE': games.label_efficient(  ),
game = games[args.game]

factor_type = args.approach.split('_')

import torch

num_devices = torch.cuda.device_count()
print('num devices', num_devices)
algos = [ neuralcbpside_v3.NeuralCBPside(game, factor_type, 1.01, 0.05,10, "cuda:{}".format(i) ) for i in range(num_devices)  ]



evaluator = Evaluation(args.game, args.task, n_folds, horizon, game, args.approach, args.context_type)


evaluate_parallel(evaluator, algos, game, id)
        
# with gzip.open( './results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach) ,'ab') as g:

#     for jobid in range(n_folds):

#         pkl.dump( result[jobid], g)











#     with gzip.open(  './results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach, jobid) ,'rb') as f:
#         r = pkl.load(f)

# bashCommand = 'rm ./results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach, jobid)
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

# algos_dico = {
#           'neuralcbp_theory':neuralcbpside_v3.NeuralCBPside(game, 'theory', 1.01, 0.05),
#           'neuralcbp_simplified':neuralcbpside_v3.NeuralCBPside(game, 'simplified', 1.01, 0.05),
#           'neuralcbp_1':neuralcbpside_v3.NeuralCBPside(game, '1', 1.01, 0.05)  }
#'CBPside':cbpside.CBPside(game, dim, factor_choice, 1.01, 0.05),