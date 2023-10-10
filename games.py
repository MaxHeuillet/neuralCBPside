from math import log, exp, pow
import numpy as np
# import geometry
import collections
import geometry_v3
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


# def apple_tasting(  ):

#     name = 'AT'
#     LossMatrix = np.array( [ [1, 0], [0, 1] ] )
#     FeedbackMatrix =  np.array([ [0, 0], [1, 2] ])
#     signal_matrices =  [ np.array( [ [1,1] ] ), np.array( [ [1,0], [0,1] ] ) ]

#     FeedbackMatrix_PMDMED =  np.array([ [0, 0],[1, 2] ])
#     A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
#     signal_matrices_Adim =  [ np.array( [ [1,1],[0,0],[0,0] ] ), np.array( [ [0,0],[1,0],[0,1] ] ) ]

#     mathcal_N = [ [0, 1] ] 

#     v = { 0: {1: [np.array([0]), np.array([1.,  -1.])]} } 

#     N_plus =  collections.defaultdict(dict)
#     N_plus[0][1] = [ 0, 1 ]

#     V = collections.defaultdict(dict)
#     V[0][1] = [ 0,1 ]

#     game = Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED,signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

#     return game


def apple_tasting(  ):

    name = 'AT'
    LossMatrix = np.array( [ [1, 0], [0, 1] ] )
    FeedbackMatrix =  np.array([ [0, 1], [2, 2] ])
    signal_matrices =  [ np.array( [ [1,0], [0,1] ] ), np.array( [ [1,1] ] ) ]

    FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [0,0],[0,0],[1,1] ] ), np.array( [ [1,0],[0,1],[0,0] ] ) ]

    mathcal_N = [ [0, 1] ] 

    v = { 0: {1: [ np.array([-1.,  1.]), np.array([0])]} } 

    N_plus =  collections.defaultdict(dict)
    N_plus[0][1] = [ 0, 1 ]

    V = collections.defaultdict(dict)
    V[0][1] = [ 0,1 ]

    game = Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED,signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

    return game




def label_efficient(  ):

    name = 'LE'
    LossMatrix = np.array( [ [1, 1],[1, 0],[0, 1] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [3, 3] ] )

    signal_matrices = [ np.array( [ [0,1],[1,0] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2],[2,2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,0],[0,1],[0,0] ] ), np.array( [ [0,0],[0,0],[1,1] ] ), np.array( [ [0,0],[0,0],[1,1] ] ) ]
    
    mathcal_N = [  [1,2] ] #,  [2,1] 

    v = {1: {2: [ np.array([-1.,  1.]), np.array([0]), np.array([0])]}, } #2: {1: [np.array([ 1., -1.]), np.array([0.]), np.array([0.])]}
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, 1, 2 ]

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )





def detection_game( threshold ):

    tho = 1/threshold -1

    name = 'DG'
    LossMatrix = np.array( [ [0, 1], [tho, 0] ] ) / np.max( game.LossMatrix )
    FeedbackMatrix =  np.array([ [0, 1], [2, 2] ])
    signal_matrices =  [ np.array( [ [1, 0], [0, 1] ] ), np.array( [ [1, 1] ] ) ]

    FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [0,0],[1,0],[0,1] ] ), np.array( [ [1,1],[0,0],[0,0] ] ) ]

    mathcal_N = [ [0, 1] ] 

    v = { 0: {1: [np.array([-tho,  1.]), np.array([0]) ]} } 

    N_plus =  collections.defaultdict(dict)
    N_plus[0][1] = [ 0, 1 ]

    V = collections.defaultdict(dict)
    V[0][1] = [ 0,1 ]

    game = Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED,signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

    return game








from scipy.optimize import fsolve
def objective_fn(b, a, T):
    return a/b - T

def solve_system(a, T):
    def objective(b):
        return objective_fn(b, a, T)

    b_opt = fsolve(objective, x0=1.0)
    return b_opt

def tho_detection( threshold ):

    name = 'tho_detection'
    a = 1
    b_opt = int( np.round( solve_system(a, threshold)[0] ) )
    LossMatrix = np.array( [ [a,a], [b_opt, 0] ] ) 
    LossMatrix = LossMatrix / np.max( LossMatrix )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2]  ] )
    signal_matrices = [ np.array( [ [1, 0], [0, 1] ]), np.array( [ [1,1] ] )  ] 


    FeedbackMatrix_PMDMED =  None
    A = None
    signal_matrices_Adim =  None

    mathcal_N = [  [0, 1], ] 

    V = collections.defaultdict(dict)
    V[1][0] = [ 0, 1 ]
    V[0][1] = [ 0, 1 ]

    N_plus =  collections.defaultdict(dict)
    N_plus[1][0] = [ 0, 1 ]
    N_plus[0][1] = [ 1, 0 ]

    # v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

    t1 = LossMatrix[0][0] - LossMatrix[1][0]
    t2 = LossMatrix[0][1] - LossMatrix[1][1]

    v = {0: {1: [ np.array([t1,  t2]), np.array([0]) ]}, } #1: {0: [ np.array([-1,  (b_opt - 1) ]), np.array([0]) ]}

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )





































def tho_detection2( threshold ):

    name = 'tho_detection2'

    b = 1
    a_opt = threshold
    LossMatrix = np.array( [ [a_opt,a_opt], [b, 0] ] ) 
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2]  ] )
    signal_matrices = [ np.array( [ [0,1], [1,0] ]), np.array( [ [1,1] ] )  ] 


    FeedbackMatrix_PMDMED =  None
    A = None
    signal_matrices_Adim =  None

    mathcal_N = [  [0, 1], ] #  [1, 0] 

    V = collections.defaultdict(dict)
    V[0][1] = [ 0, 1 ]

    N_plus =  collections.defaultdict(dict)
    N_plus[0][1] = [ 1, 0 ]

    # v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

    v = {0: {1: [ np.array([1.,  -(b - 1)]), np.array([0]) ]}, }

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )
