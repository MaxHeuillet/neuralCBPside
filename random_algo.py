

import numpy as np


class Random():

    def __init__(self, game,):
        self.name = 'random'
        self.game = game
        self.N = game.n_actions

    def get_action(self, t, context = None ):
        
        pbt = np.ones( self.game.n_actions ) / self.game.n_actions
        action = np.random.choice(self.game.n_actions, 1,  p = pbt )[0]
        explored = 1 if action == 0 else 0
        

        history = {'monitor_action':action, 'explore':explored,}
            
        return action, history

    def update(self, action, feedback, outcome, context, t):
        return None, None

    def reset(self, d):
        pass
