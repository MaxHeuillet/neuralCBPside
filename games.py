from math import log, exp, pow
import numpy as np
# import geometry
import collections
# import geometry_v3
from itertools import combinations, permutations
import random

class Game():
    
     def __init__(self, name, informative_symbols, noise, LossMatrix, FeedbackMatrix, SignalMatrices, mathcal_N, v, N_plus, V ):
        
        self.name = name
        self.LossMatrix = LossMatrix
        self.FeedbackMatrix = FeedbackMatrix
        self.SignalMatrices = SignalMatrices
        self.n_actions = len(self.LossMatrix)
        self.n_outcomes = len(self.LossMatrix[0])
        self.mathcal_N = mathcal_N 
        self.v = v
        self.N_plus = N_plus
        self.V = V


        self.N = len(self.LossMatrix)
        self.M = len(self.LossMatrix[0])
        self.noise = noise
        self.informative_symbols = informative_symbols
        
     def get_feedback(self, action, outcome ):
          if self.noise.get(action) == None:
               feedback = self.FeedbackMatrix[ action ][ outcome ]
          else:
               noise = self.noise[action]
               feedbacks = self.FeedbackMatrix[ action ]
               feedback = random.choices(feedbacks, weights=noise)[0]
          return feedback

def game_bandit( noise ):

    name = 'bandit'
    LossMatrix = np.array( [ [1, 1],[0, 1],[1, 0] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [2, 2] ] )

    signal_matrices = [ np.array( [ [1, 0],[0, 1] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    mathcal_N = [  [1,2] ] 

    v = {1: {2: [ np.array([-1,  1]), np.array([0]), np.array([0])]}, } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, ]

    informative_symbols = [0, 1]

    return Game( name, informative_symbols, noise, LossMatrix, FeedbackMatrix, signal_matrices, mathcal_N, v, N_plus, V )



def game_case1( noise ):

    name = 'case1'
    LossMatrix = np.array( [ [1, 1],[0, 1],[1, 0] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [2, 2] ] )

    signal_matrices = [ np.array( [ [1, 0],[0, 1] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    mathcal_N = [  [1,2] ] 

    v = {1: {2: [ np.array([-1,  1]), np.array([0]), np.array([0])]}, } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, ]

    informative_symbols = [0, 1]

    return Game( name, informative_symbols, noise, LossMatrix, FeedbackMatrix, signal_matrices, mathcal_N, v, N_plus, V )


def game_case1b( noise ):

    name = 'case1b'
    LossMatrix = np.array( [ [1, 1],[0, 1],[1/2, 0] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [2, 2] ] )

    signal_matrices = [ np.array( [ [1, 0],[0, 1] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    mathcal_N = [  [1,2] ] 

    v = {1: {2: [ np.array([-1/2,  1]) ]}, } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, ]

    informative_symbols = [0, 1]

    return Game( name, informative_symbols, noise, LossMatrix, FeedbackMatrix, signal_matrices, mathcal_N, v, N_plus, V )




def game_case2( noise ):

    name = 'case2'
    LossMatrix = np.array( [ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ] )
    
    FeedbackMatrix = np.array(  [ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                  [10] * 10, 
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10,
                                  [10] * 10 ] )

    signal_matrices = [ np.array( [ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ] ), 

                                    np.array( [ [1]*10 ] ), 
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] ),
                                    np.array( [ [1]*10 ] )  ] 

    
    mathcal_N = [  [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [1,10],
                   [2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9], [2,10],
                   [3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10],
                   [4,5], [4,6], [4,7], [4,8], [4,9], [4,10],
                   [5,6], [5,7], [5,8], [5,9], [5,10],
                   [6,7], [6,8], [6,9], [6,10],
                   [7,8], [7,9], [7,10],
                   [8,9], [8,10],
                   [9,10],  ]   

    v = {1: {2: [ np.array([-1,  1, 0, 0, 0, 0, 0, 0, 0, 0]) ],
             3: [ np.array([-1,  0, 1, 0, 0, 0, 0, 0, 0, 0]) ],
             4: [ np.array([-1,  0, 0, 1, 0, 0, 0, 0, 0, 0]) ],
             5: [ np.array([-1,  0, 0, 0, 1, 0, 0, 0, 0, 0]) ],
             6: [ np.array([-1,  0, 0, 0, 0, 1, 0, 0, 0, 0]) ],
             7: [ np.array([-1,  0, 0, 0, 0, 0, 1, 0, 0, 0]) ],
             8: [ np.array([-1,  0, 0, 0, 0, 0, 0, 1, 0, 0]) ],
             9: [ np.array([-1,  0, 0, 0, 0, 0, 0, 0, 1, 0]) ],
             10:[ np.array([-1,  0, 0, 0, 0, 0, 0, 0, 0, 1]) ] },

        2: { 3: [ np.array([0,  -1, 1, 0, 0, 0, 0, 0, 0, 0]) ],
             4: [ np.array([0,  -1, 0, 1, 0, 0, 0, 0, 0, 0]) ],
             5: [ np.array([0,  -1, 0, 0, 1, 0, 0, 0, 0, 0]) ],
             6: [ np.array([0,  -1, 0, 0, 0, 1, 0, 0, 0, 0]) ],
             7: [ np.array([0,  -1, 0, 0, 0, 0, 1, 0, 0, 0]) ],
             8: [ np.array([0,  -1, 0, 0, 0, 0, 0, 1, 0, 0]) ],
             9: [ np.array([0,  -1, 0, 0, 0, 0, 0, 0, 1, 0]) ],
             10:[ np.array([0,  -1, 0, 0, 0, 0, 0, 0, 0, 1]) ] },

        3: { 4: [ np.array([0,  0, -1, 1, 0, 0, 0, 0, 0, 0]) ],
             5: [ np.array([0,  0, -1, 0, 1, 0, 0, 0, 0, 0]) ],
             6: [ np.array([0,  0, -1, 0, 0, 1, 0, 0, 0, 0]) ],
             7: [ np.array([0,  0, -1, 0, 0, 0, 1, 0, 0, 0]) ],
             8: [ np.array([0,  0, -1, 0, 0, 0, 0, 1, 0, 0]) ],
             9: [ np.array([0,  0, -1, 0, 0, 0, 0, 0, 1, 0]) ],
             10:[ np.array([0,  0, -1, 0, 0, 0, 0, 0, 0, 1]) ] },

        4: { 5: [ np.array([0,  0, 0, -1, 1, 0, 0, 0, 0, 0]) ],
             6: [ np.array([0,  0, 0, -1, 0, 1, 0, 0, 0, 0]) ],
             7: [ np.array([0,  0, 0, -1, 0, 0, 1, 0, 0, 0]) ],
             8: [ np.array([0,  0, 0, -1, 0, 0, 0, 1,  0, 0]) ],
             9: [ np.array([0,  0, 0, -1, 0, 0, 0, 0, 1, 0]) ],
             10:[ np.array([0,  0, 0, -1, 0, 0, 0, 0, 0, 1]) ] },

        5: { 6: [ np.array([0,  0, 0, 0, -1, 1, 0, 0, 0, 0]) ],
             7: [ np.array([0,  0, 0, 0, -1, 0, 1, 0, 0, 0]) ],
             8: [ np.array([0,  0, 0, 0, -1, 0, 0, 1, 0, 0]) ],
             9: [ np.array([0,  0, 0, 0, -1, 0, 0, 0, 1, 0]) ],
             10:[ np.array([0,  0, 0, 0, -1, 0, 0, 0, 0, 1]) ] },

        6: { 7: [ np.array([0,  0, 0, 0, 0, -1, 1, 0, 0, 0]) ],
             8: [ np.array([0,  0, 0, 0, 0, -1, 0, 1, 0, 0]) ],
             9: [ np.array([0,  0, 0, 0, 0, -1, 0, 0, 1, 0]) ],
             10:[ np.array([0,  0, 0, 0, 0, -1, 0, 0, 0, 1]) ] },

        7: { 8: [ np.array([0,  0, 0, 0, 0, 0, -1, 1, 0, 0]) ],
             9: [ np.array([0,  0, 0, 0, 0, 0, -1, 0, 1, 0]) ],
             10:[ np.array([0, 0, 0, 0, 0, 0,  -1, 0, 0, 1]) ] },

        8: { 9: [ np.array([0,  0, 0, 0, 0, 0, 0, -1, 1, 0]) ],
             10:[ np.array([0, 0, 0, 0, 0, 0, 0,  -1,  0, 1]) ] },

        9: { 10:[ np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 1]) ] },  } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]
    N_plus[1][3] = [ 1, 3 ]
    N_plus[1][4] = [ 1, 4 ]
    N_plus[1][5] = [ 1, 5 ]
    N_plus[1][6] = [ 1, 6 ]
    N_plus[1][7] = [ 1, 7 ]
    N_plus[1][8] = [ 1, 8 ]
    N_plus[1][9] = [ 1, 9 ]
    N_plus[1][10] = [1, 10 ]

    N_plus[2][3] = [ 2, 3 ]
    N_plus[2][4] = [ 2, 4 ]
    N_plus[2][5] = [ 2, 5 ]
    N_plus[2][6] = [ 2, 6 ]
    N_plus[2][7] = [ 2, 7 ]
    N_plus[2][8] = [ 2, 8 ]
    N_plus[2][9] = [ 2, 9 ]
    N_plus[2][10] = [2, 10 ]

    N_plus[3][4] = [ 3, 4 ]
    N_plus[3][5] = [ 3, 5 ]
    N_plus[3][6] = [ 3, 6 ]
    N_plus[3][7] = [ 3, 7 ]
    N_plus[3][8] = [ 3, 8 ]
    N_plus[3][9] = [ 3, 9 ]
    N_plus[3][10] = [3, 10]

    N_plus[4][5] = [ 4, 5 ]
    N_plus[4][6] = [ 4, 6 ]
    N_plus[4][7] = [ 4, 7 ]
    N_plus[4][8] = [ 4, 8 ]
    N_plus[4][9] = [ 4, 9 ]
    N_plus[4][10] = [4, 10 ]

    N_plus[5][6] = [ 5, 6 ]
    N_plus[5][7] = [ 5, 7 ]
    N_plus[5][8] = [ 5, 8 ]
    N_plus[5][9] = [ 5, 9 ]
    N_plus[5][10] = [5, 10 ]

    N_plus[6][7] = [ 6, 7 ]
    N_plus[6][8] = [ 6, 8 ]
    N_plus[6][9] = [ 6, 9 ]
    N_plus[6][10] = [6, 10 ]

    N_plus[7][8] = [ 7, 8 ]
    N_plus[7][9] = [ 7, 9 ]
    N_plus[7][10] = [7, 10 ]

    N_plus[8][9] = [ 8, 9 ]
    N_plus[8][10] = [8, 10 ]

    N_plus[9][10] = [ 9, 10 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, ] 
    V[1][3] = [ 0, ]  
    V[1][4] = [ 0, ]  
    V[1][5] = [ 0, ]  
    V[1][6] = [ 0, ]  
    V[1][7] = [ 0, ] 
    V[1][8] = [ 0, ]  
    V[1][9] = [ 0, ]  
    V[1][10] = [0, ]  

    V[2][3] = [ 0, ]
    V[2][4] = [ 0, ]
    V[2][5] = [ 0, ]
    V[2][6] = [ 0, ]
    V[2][7] = [ 0, ]
    V[2][8] = [ 0, ]
    V[2][9] = [ 0, ]
    V[2][10] = [0, ]

    V[3][4] = [ 0, ]
    V[3][5] = [ 0, ]
    V[3][6] = [ 0, ]
    V[3][7] = [ 0, ]
    V[3][8] = [ 0, ]
    V[3][9] = [ 0, ]
    V[3][10] = [0,  ]

    V[4][5] = [ 0, ]
    V[4][6] = [ 0, ]
    V[4][7] = [ 0, ]
    V[4][8] = [ 0, ]
    V[4][9] = [ 0, ]
    V[4][10] = [0,  ]

    V[5][6] = [ 0, ]
    V[5][7] = [ 0, ]
    V[5][8] = [ 0, ]
    V[5][9] = [ 0, ]
    V[5][10] = [0,  ]

    V[6][7] = [ 0, ]
    V[6][8] = [ 0, ]
    V[6][9] = [ 0, ]
    V[6][10] = [0,  ]

    V[7][8] = [ 0, ]
    V[7][9] = [ 0, ]
    V[7][10] = [0,  ]

    V[8][9] = [ 0, ]
    V[8][10] = [0,  ]

    V[9][10] = [ 0,  ]

    informative_symbols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]

    return Game( name,  informative_symbols, noise, LossMatrix, FeedbackMatrix, signal_matrices, mathcal_N, v, N_plus, V )



def game_case_seven( noise ):

    name = 'case_seven'
    LossMatrix = np.array( [ [1, 1, 1, 1, 1, 1, 1, ],
                             [0, 1, 1, 1, 1, 1, 1, ],
                             [1, 0, 1, 1, 1, 1, 1, ],
                             [1, 1, 0, 1, 1, 1, 1, ],
                             [1, 1, 1, 0, 1, 1, 1, ],
                             [1, 1, 1, 1, 0, 1, 1, ],
                             [1, 1, 1, 1, 1, 0, 1, ],
                             [1, 1, 1, 1, 1, 1, 0, ], ] )
    
    FeedbackMatrix = np.array(  [ [0, 1, 2, 3, 4, 5, 6, ], 
                                  [10] * 7, 
                                  [10] * 7,
                                  [10] * 7,
                                  [10] * 7,
                                  [10] * 7,
                                  [10] * 7,
                                  [10] * 7, ] )

    signal_matrices = [ np.array( [ [1, 0, 0, 0, 0, 0, 0,],
                                    [0, 1, 0, 0, 0, 0, 0,],
                                    [0, 0, 1, 0, 0, 0, 0,],
                                    [0, 0, 0, 1, 0, 0, 0,],
                                    [0, 0, 0, 0, 1, 0, 0,],
                                    [0, 0, 0, 0, 0, 1, 0,],
                                    [0, 0, 0, 0, 0, 0, 1,], ] ), 

                                    np.array( [ [1]*7 ] ), 
                                    np.array( [ [1]*7 ] ),
                                    np.array( [ [1]*7 ] ),
                                    np.array( [ [1]*7 ] ),
                                    np.array( [ [1]*7 ] ),
                                    np.array( [ [1]*7 ] ),
                                    np.array( [ [1]*7 ] ), ] 

    
    mathcal_N = [  [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], 
                   [2,3], [2,4], [2,5], [2,6], [2,7], 
                   [3,4], [3,5], [3,6], [3,7], 
                   [4,5], [4,6], [4,7], 
                   [5,6], [5,7], 
                   [6,7],   ]   

    v = {1: {2: [ np.array([-1,  1, 0, 0, 0, 0, 0, ]) ],
             3: [ np.array([-1,  0, 1, 0, 0, 0, 0, ]) ],
             4: [ np.array([-1,  0, 0, 1, 0, 0, 0, ]) ],
             5: [ np.array([-1,  0, 0, 0, 1, 0, 0, ]) ],
             6: [ np.array([-1,  0, 0, 0, 0, 1, 0, ]) ],
             7: [ np.array([-1,  0, 0, 0, 0, 0, 1, ]) ], },

        2: { 3: [ np.array([0,  -1, 1, 0, 0, 0, 0, ]) ],
             4: [ np.array([0,  -1, 0, 1, 0, 0, 0, ]) ],
             5: [ np.array([0,  -1, 0, 0, 1, 0, 0, ]) ],
             6: [ np.array([0,  -1, 0, 0, 0, 1, 0, ]) ],
             7: [ np.array([0,  -1, 0, 0, 0, 0, 1, ]) ], },

        3: { 4: [ np.array([0,  0, -1, 1, 0, 0, 0,]) ],
             5: [ np.array([0,  0, -1, 0, 1, 0, 0,]) ],
             6: [ np.array([0,  0, -1, 0, 0, 1, 0,]) ],
             7: [ np.array([0,  0, -1, 0, 0, 0, 1,]) ],},

        4: { 5: [ np.array([0,  0, 0, -1, 1, 0, 0, ]) ],
             6: [ np.array([0,  0, 0, -1, 0, 1, 0, ]) ],
             7: [ np.array([0,  0, 0, -1, 0, 0, 1, ]) ], },

        5: { 6: [ np.array([0,  0, 0, 0, -1, 1, 0, ]) ],
             7: [ np.array([0,  0, 0, 0, -1, 0, 1, ]) ],},

        6: { 7: [ np.array([0,  0, 0, 0, 0, -1, 1, ]) ],},  } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]
    N_plus[1][3] = [ 1, 3 ]
    N_plus[1][4] = [ 1, 4 ]
    N_plus[1][5] = [ 1, 5 ]
    N_plus[1][6] = [ 1, 6 ]
    N_plus[1][7] = [ 1, 7 ]

    N_plus[2][3] = [ 2, 3 ]
    N_plus[2][4] = [ 2, 4 ]
    N_plus[2][5] = [ 2, 5 ]
    N_plus[2][6] = [ 2, 6 ]
    N_plus[2][7] = [ 2, 7 ]

    N_plus[3][4] = [ 3, 4 ]
    N_plus[3][5] = [ 3, 5 ]
    N_plus[3][6] = [ 3, 6 ]
    N_plus[3][7] = [ 3, 7 ]

    N_plus[4][5] = [ 4, 5 ]
    N_plus[4][6] = [ 4, 6 ]
    N_plus[4][7] = [ 4, 7 ]

    N_plus[5][6] = [ 5, 6 ]
    N_plus[5][7] = [ 5, 7 ]

    N_plus[6][7] = [ 6, 7 ]


    V = collections.defaultdict(dict)
    V[1][2] = [ 0,  ] 
    V[1][3] = [ 0, ]  
    V[1][4] = [ 0, ]  
    V[1][5] = [ 0, ]  
    V[1][6] = [ 0, ]  
    V[1][7] = [ 0, ] 
    
    V[2][3] = [ 0, ]
    V[2][4] = [ 0, ]
    V[2][5] = [ 0, ]
    V[2][6] = [ 0, ]
    V[2][7] = [ 0, ]

    V[3][4] = [ 0, ]
    V[3][5] = [ 0, ]
    V[3][6] = [ 0, ]
    V[3][7] = [ 0, ]

    V[4][5] = [ 0, ]
    V[4][6] = [ 0, ]
    V[4][7] = [ 0, ]

    V[5][6] = [ 0, ]
    V[5][7] = [ 0, ]

    V[6][7] = [ 0, ]

    informative_symbols = [0, 1, 2, 3, 4, 5, 6, ]

    return Game( name,  informative_symbols, noise, LossMatrix, FeedbackMatrix, signal_matrices, mathcal_N, v, N_plus, V )










