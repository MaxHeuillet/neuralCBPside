
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

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
        action_history = [None] * self.horizon
        outcome_history = [None] * self.horizon
        pred_performance = {}
        n_verifs = 0

        for t in range(self.horizon):

            if t % 1000 == 0 :
                print(t)

            context, distribution = context_generator.get_context()
            print(context.shape)


            outcome = np.argmax(distribution) 

            
            action, _ = alg.get_action(t, context)


            feedback =  self.get_feedback( game, action, outcome )
            

            alg.update(action, feedback, outcome, t, context, game.LossMatrix )

            if action == 0:
                n_verifs += 1

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            cumRegret[t] =  val
            action_history[t] = action
            outcome_history[t] = outcome
            print('t', t, 'action', action, 'outcome', outcome, 'regret', val  )

            if n_verifs in [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 9000] and n_verifs not in pred_performance.keys():
            # if t in [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 9000]:
                X, y = context_generator.get_test_data()
                X = X.to('cuda:0')
                y_probas = alg.predictor(X,y)
                y_pred = torch.argmax( y_probas, 1 ).tolist()
                acc = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average='weighted')
                pred_performance[n_verifs] = {'accuracy':acc, 'f1':f1, 'nverifs':n_verifs}

        result = {'regret': np.cumsum(cumRegret), 'action_history':action_history,
                  'outcome_history':outcome_history, 'pred':pred_performance}
        print(result)
        print('finished')
        with gzip.open( './results/{}_{}_{}_{}_{}_{}.pkl.gz'.format(self.case, self.model, self.context_type, self.horizon, self.n_folds, self.label) ,'ab') as f:
            pkl.dump(result,f)
        print('saved')

        return result