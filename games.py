from math import log, exp, pow
import numpy as np
# import geometry
import collections
# import geometry_v3
from itertools import combinations, permutations

class Game():
    
    def __init__(self, name, LossMatrix, FeedbackMatrix,FeedbackMatrix_PMDMED, SignalMatrices, signal_matrices_Adim, mathcal_N, v, N_plus, V ):
        
        self.name = name
        self.LossMatrix = LossMatrix
        self.FeedbackMatrix = FeedbackMatrix
        self.FeedbackMatrix_PMDMED = FeedbackMatrix_PMDMED

        self.SignalMatrices = SignalMatrices
        self.SignalMatricesAdim = signal_matrices_Adim
        self.n_actions = len(self.LossMatrix)
        self.n_outcomes = len(self.LossMatrix[0])
        self.mathcal_N = mathcal_N 
        self.v = v
        self.N_plus = N_plus
        self.V = V

        self.N = len(self.LossMatrix)
        self.M = len(self.LossMatrix[0])

def game_case1(  ):

    name = 'case1'
    LossMatrix = np.array( [ [1, 1],[1, 0],[0, 1] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [2, 2] ] )

    signal_matrices = [ np.array( [ [0,1],[1,0] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2],[2,2] ])
    A = None #geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,0],[0,1],[0,0] ] ), np.array( [ [0,0],[0,0],[1,1] ] ), np.array( [ [0,0],[0,0],[1,1] ] ) ]
    
    mathcal_N = [  [1,2] ] 

    v = {1: {2: [ np.array([-1.,  1.]), np.array([0]), np.array([0])]}, } 
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, 1, 2 ]

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )




def game_case2(  ):

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

    FeedbackMatrix_PMDMED =  FeedbackMatrix.copy()
    A = None #geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  None
    
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
    V[1][2] = [ 0,  ] 
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
    V[2][10] = [0,  ]

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

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )









def game_case3(  ):

    name = 'case3'
    LossMatrix = np.array( [ [2, 1, 3, 1, 2, 1, 10, 1, 1, 1],
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
                                  [-1] * 10, 
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10,
                                  [-1] * 10 ] )

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

    FeedbackMatrix_PMDMED =  FeedbackMatrix.copy()
    A = None #geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  None
    
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
    V[1][2] = [ 0,  ] 
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
    V[2][10] = [0,  ]

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

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )













# def game_case4(  ):

#     name = 'case4'
#     LossMatrix = np.array( [ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
#                              [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
#                              [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
#                              [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
#                              [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#                              [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#                              [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
#                              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
#                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] ] )
    
#     FeedbackMatrix = np.array(  [ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
#                                   [-1] * 10, 
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10,
#                                   [-1] * 10 ] )

#     signal_matrices = [ np.array( [ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ] ), 

#                                     np.array( [ [1]*10 ] ), 
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] ),
#                                     np.array( [ [1]*10 ] )  ] 

#     FeedbackMatrix_PMDMED =  FeedbackMatrix.copy()
#     A = None #geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
#     signal_matrices_Adim =  None
    
#     mathcal_N = [  [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [1,10],
#                    [2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9], [2,10],
#                    [3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10],
#                    [4,5], [4,6], [4,7], [4,8], [4,9], [4,10],
#                    [5,6], [5,7], [5,8], [5,9], [5,10],
#                    [6,7], [6,8], [6,9], [6,10],
#                    [7,8], [7,9], [7,10],
#                    [8,9], [8,10],
#                    [9,10],  ]   

#     v = {1: {2: [ np.array([-1,  1, 0, 0, 0, 0, 0, 0, 0, 0]) ],
#              3: [ np.array([-1,  0, 1, 0, 0, 0, 0, 0, 0, 0]) ],
#              4: [ np.array([-1,  0, 0, 1, 0, 0, 0, 0, 0, 0]) ],
#              5: [ np.array([-1,  0, 0, 0, 1, 0, 0, 0, 0, 0]) ],
#              6: [ np.array([-1,  0, 0, 0, 0, 1, 0, 0, 0, 0]) ],
#              7: [ np.array([-1,  0, 0, 0, 0, 0, 1, 0, 0, 0]) ],
#              8: [ np.array([-1,  0, 0, 0, 0, 0, 0, 1, 0, 0]) ],
#              9: [ np.array([-1,  0, 0, 0, 0, 0, 0, 0, 1, 0]) ],
#              10:[ np.array([-1,  0, 0, 0, 0, 0, 0, 0, 0, 1]) ] },

#         2: { 3: [ np.array([0,  -1, 1, 0, 0, 0, 0, 0, 0, 0]) ],
#              4: [ np.array([0,  -1, 0, 1, 0, 0, 0, 0, 0, 0]) ],
#              5: [ np.array([0,  -1, 0, 0, 1, 0, 0, 0, 0, 0]) ],
#              6: [ np.array([0,  -1, 0, 0, 0, 1, 0, 0, 0, 0]) ],
#              7: [ np.array([0,  -1, 0, 0, 0, 0, 1, 0, 0, 0]) ],
#              8: [ np.array([0,  -1, 0, 0, 0, 0, 0, 1, 0, 0]) ],
#              9: [ np.array([0,  -1, 0, 0, 0, 0, 0, 0, 1, 0]) ],
#              10:[ np.array([0,  -1, 0, 0, 0, 0, 0, 0, 0, 1]) ] },

#         3: { 4: [ np.array([0,  0, -1, 1, 0, 0, 0, 0, 0, 0]) ],
#              5: [ np.array([0,  0, -1, 0, 1, 0, 0, 0, 0, 0]) ],
#              6: [ np.array([0,  0, -1, 0, 0, 1, 0, 0, 0, 0]) ],
#              7: [ np.array([0,  0, -1, 0, 0, 0, 1, 0, 0, 0]) ],
#              8: [ np.array([0,  0, -1, 0, 0, 0, 0, 1, 0, 0]) ],
#              9: [ np.array([0,  0, -1, 0, 0, 0, 0, 0, 1, 0]) ],
#              10:[ np.array([0,  0, -1, 0, 0, 0, 0, 0, 0, 1]) ] },

#         4: { 5: [ np.array([0,  0, 0, -1, 1, 0, 0, 0, 0, 0]) ],
#              6: [ np.array([0,  0, 0, -1, 0, 1, 0, 0, 0, 0]) ],
#              7: [ np.array([0,  0, 0, -1, 0, 0, 1, 0, 0, 0]) ],
#              8: [ np.array([0,  0, 0, -1, 0, 0, 0, 1,  0, 0]) ],
#              9: [ np.array([0,  0, 0, -1, 0, 0, 0, 0, 1, 0]) ],
#              10:[ np.array([0,  0, 0, -1, 0, 0, 0, 0, 0, 1]) ] },

#         5: { 6: [ np.array([0,  0, 0, 0, -1, 1, 0, 0, 0, 0]) ],
#              7: [ np.array([0,  0, 0, 0, -1, 0, 1, 0, 0, 0]) ],
#              8: [ np.array([0,  0, 0, 0, -1, 0, 0, 1, 0, 0]) ],
#              9: [ np.array([0,  0, 0, 0, -1, 0, 0, 0, 1, 0]) ],
#              10:[ np.array([0,  0, 0, 0, -1, 0, 0, 0, 0, 1]) ] },

#         6: { 7: [ np.array([0,  0, 0, 0, 0, -1, 1, 0, 0, 0]) ],
#              8: [ np.array([0,  0, 0, 0, 0, -1, 0, 1, 0, 0]) ],
#              9: [ np.array([0,  0, 0, 0, 0, -1, 0, 0, 1, 0]) ],
#              10:[ np.array([0,  0, 0, 0, 0, -1, 0, 0, 0, 1]) ] },

#         7: { 8: [ np.array([0,  0, 0, 0, 0, 0, -1, 1, 0, 0]) ],
#              9: [ np.array([0,  0, 0, 0, 0, 0, -1, 0, 1, 0]) ],
#              10:[ np.array([0, 0, 0, 0, 0, 0,  -1, 0, 0, 1]) ] },

#         8: { 9: [ np.array([0,  0, 0, 0, 0, 0, 0, -1, 1, 0]) ],
#              10:[ np.array([0, 0, 0, 0, 0, 0, 0,  -1,  0, 1]) ] },

#         9: { 10:[ np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 1]) ] },  } 
    
#     N_plus =  collections.defaultdict(dict)
#     N_plus[1][2] = [ 1, 2 ]
#     N_plus[1][3] = [ 1, 3 ]
#     N_plus[1][4] = [ 1, 4 ]
#     N_plus[1][5] = [ 1, 5 ]
#     N_plus[1][6] = [ 1, 6 ]
#     N_plus[1][7] = [ 1, 7 ]
#     N_plus[1][8] = [ 1, 8 ]
#     N_plus[1][9] = [ 1, 9 ]
#     N_plus[1][10] = [1, 10 ]

#     N_plus[2][3] = [ 2, 3 ]
#     N_plus[2][4] = [ 2, 4 ]
#     N_plus[2][5] = [ 2, 5 ]
#     N_plus[2][6] = [ 2, 6 ]
#     N_plus[2][7] = [ 2, 7 ]
#     N_plus[2][8] = [ 2, 8 ]
#     N_plus[2][9] = [ 2, 9 ]
#     N_plus[2][10] = [2, 10 ]

#     N_plus[3][4] = [ 3, 4 ]
#     N_plus[3][5] = [ 3, 5 ]
#     N_plus[3][6] = [ 3, 6 ]
#     N_plus[3][7] = [ 3, 7 ]
#     N_plus[3][8] = [ 3, 8 ]
#     N_plus[3][9] = [ 3, 9 ]
#     N_plus[3][10] = [3, 10]

#     N_plus[4][5] = [ 4, 5 ]
#     N_plus[4][6] = [ 4, 6 ]
#     N_plus[4][7] = [ 4, 7 ]
#     N_plus[4][8] = [ 4, 8 ]
#     N_plus[4][9] = [ 4, 9 ]
#     N_plus[4][10] = [4, 10 ]

#     N_plus[5][6] = [ 5, 6 ]
#     N_plus[5][7] = [ 5, 7 ]
#     N_plus[5][8] = [ 5, 8 ]
#     N_plus[5][9] = [ 5, 9 ]
#     N_plus[5][10] = [5, 10 ]

#     N_plus[6][7] = [ 6, 7 ]
#     N_plus[6][8] = [ 6, 8 ]
#     N_plus[6][9] = [ 6, 9 ]
#     N_plus[6][10] = [6, 10 ]

#     N_plus[7][8] = [ 7, 8 ]
#     N_plus[7][9] = [ 7, 9 ]
#     N_plus[7][10] = [7, 10 ]

#     N_plus[8][9] = [ 8, 9 ]
#     N_plus[8][10] = [8, 10 ]

#     N_plus[9][10] = [ 9, 10 ]

#     V = collections.defaultdict(dict)
#     V[1][2] = [ 0,  ] 
#     V[1][3] = [ 0, ]  
#     V[1][4] = [ 0, ]  
#     V[1][5] = [ 0, ]  
#     V[1][6] = [ 0, ]  
#     V[1][7] = [ 0, ] 
#     V[1][8] = [ 0, ]  
#     V[1][9] = [ 0, ]  
#     V[1][10] = [0, ]  

#     V[2][3] = [ 0, ]
#     V[2][4] = [ 0, ]
#     V[2][5] = [ 0, ]
#     V[2][6] = [ 0, ]
#     V[2][7] = [ 0, ]
#     V[2][8] = [ 0, ]
#     V[2][9] = [ 0, ]
#     V[2][10] = [0,  ]

#     V[3][4] = [ 0, ]
#     V[3][5] = [ 0, ]
#     V[3][6] = [ 0, ]
#     V[3][7] = [ 0, ]
#     V[3][8] = [ 0, ]
#     V[3][9] = [ 0, ]
#     V[3][10] = [0,  ]

#     V[4][5] = [ 0, ]
#     V[4][6] = [ 0, ]
#     V[4][7] = [ 0, ]
#     V[4][8] = [ 0, ]
#     V[4][9] = [ 0, ]
#     V[4][10] = [0,  ]

#     V[5][6] = [ 0, ]
#     V[5][7] = [ 0, ]
#     V[5][8] = [ 0, ]
#     V[5][9] = [ 0, ]
#     V[5][10] = [0,  ]

#     V[6][7] = [ 0, ]
#     V[6][8] = [ 0, ]
#     V[6][9] = [ 0, ]
#     V[6][10] = [0,  ]

#     V[7][8] = [ 0, ]
#     V[7][9] = [ 0, ]
#     V[7][10] = [0,  ]

#     V[8][9] = [ 0, ]
#     V[8][10] = [0,  ]

#     V[9][10] = [ 0,  ]

#     return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )



































# def tho_detection2( threshold ):

#     name = 'tho_detection2'

#     b = 1
#     a_opt = threshold
#     LossMatrix = np.array( [ [a_opt,a_opt], [b, 0] ] ) 
#     FeedbackMatrix = np.array(  [ [0, 1], [2, 2]  ] )
#     signal_matrices = [ np.array( [ [0,1], [1,0] ]), np.array( [ [1,1] ] )  ] 


#     FeedbackMatrix_PMDMED =  None
#     A = None
#     signal_matrices_Adim =  None

#     mathcal_N = [  [0, 1], ] #  [1, 0] 

#     V = collections.defaultdict(dict)
#     V[0][1] = [ 0, 1 ]

#     N_plus =  collections.defaultdict(dict)
#     N_plus[0][1] = [ 1, 0 ]

#     # v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

#     v = {0: {1: [ np.array([1.,  -(b - 1)]), np.array([0]) ]}, }

#     return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )
