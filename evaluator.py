
import numpy as np
import os
from functools import partial
import pickle as pkl
import gzip
import argparse
import os
import torch
import random

from functools import partial
import pickle as pkl
import gzip

class Evaluation:

    def __init__(self, case, model, n_folds, horizon, game, label, context_type):

        self.model = model
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
    
    def set_random_seeds(self, seed):
        #print('start 2', alg.device)
        random.seed(seed)

        self.agent_random_state = np.random.RandomState(seed)
        self.agent_random_state.seed(seed)

        self.env_random_state = np.random.RandomState(seed)
        self.env_random_state.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def eval_policy_once(self, game, job):

        # print('start 1')
        context_generator, alg = job

        cumRegret =  np.zeros(self.horizon, dtype =float)
        print('start 3')

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context, distribution = context_generator.get_context()
            print(context.shape)


            # print('context', context)
            if self.game.M>2:
                outcome = np.argmax(distribution) 
            else:
                outcome = 0 if distribution[0]>0.5 else 1

            
            #print('context shape', context.shape)
            
            action, _ = alg.get_action(t, context)

            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            cumRegret[t] =  val

            print('t', t, 'action', action, 'outcome', outcome, 'regret', val  )


        result = np.cumsum(cumRegret)
        print(result)
        print('finished')
        with gzip.open( './results/{}_{}_{}_{}_{}_{}.pkl.gz'.format(self.case, self.model, self.context_type, self.horizon, self.n_folds, self.label) ,'ab') as f:
            pkl.dump(result,f)
        print('saved')

        return True