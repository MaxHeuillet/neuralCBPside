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


def apple_tasting(  ):

    name = 'AT'
    LossMatrix = np.array( [ [1, 0], [0, 1] ] )
    FeedbackMatrix =  np.array([ [0, 0], [1, 2] ])
    signal_matrices =  [ np.array( [ [1,1] ] ), np.array( [ [1,0], [0,1] ] ) ]

    FeedbackMatrix_PMDMED =  np.array([ [0, 0],[1, 2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,1],[0,0],[0,0] ] ), np.array( [ [0,0],[1,0],[0,1] ] ) ]

    mathcal_N = [ [0, 1] ] 

    v = { 0: {1: [np.array([0]), np.array([1.,  -1.])]} } 

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
    N_plus[2][1] = [ 1, 2 ]
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[2][1] = [ 0, 1, 2 ]
    V[1][2] = [ 0, 1, 2 ]

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )


# from scipy.optimize import fsolve

# def objective_fn(b, a, T):
#     return a/b - T

# def solve_system(a, T):
#     def objective(b):
#         return objective_fn(b, a, T)

#     b_opt = fsolve(objective, x0=1.0)
#     return b_opt

# def label_efficient2( threshold ):

#     name = 'LE2'
#     a = 1
#     b_opt = int( np.round( solve_system(a, threshold)[0] ) )
#     LossMatrix = np.array( [ [a,a], [b_opt, 0] ] )
#     FeedbackMatrix = np.array(  [ [1, 0], [1, 1]  ] )
#     signal_matrices = [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ])  ] 

#     bandit_LossMatrix = None
#     bandit_FeedbackMatrix =  None

#     FeedbackMatrix_PMDMED =  None
#     A = None
#     signal_matrices_Adim =  None

#     mathcal_N = [  [0, 1],  [1, 0] ] 

#     V = collections.defaultdict(dict)
#     V[1][0] = [ 0, 1 ]
#     V[0][1] = [ 0, 1 ]

#     N_plus =  collections.defaultdict(dict)
#     N_plus[1][0] = [ 0, 1 ]
#     N_plus[0][1] = [ 1, 0 ]

#     LossMatrix = np.array( [ [1, 0], [0.5, 0.5] ] )
#     FeedbackMatrix = np.array(  [ [1, 0], [1, 1]  ] )
#     LinkMatrix = None
#     signal_matrices = [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ])  ] 

#     bandit_LossMatrix = None
#     bandit_FeedbackMatrix =  None

#     FeedbackMatrix_PMDMED =  None
#     A = None
#     signal_matrices_Adim =  None
    
#     mathcal_N = [  [0, 1],  [1, 0] ] 

#     v = {0: {1: {0: np.array([ 0.5, -0.5]), 1: np.array([0.])}},
#              1: {0: {0: np.array([-0.5,  0.5]), 1: np.array([0.])}}} 

#     N_plus =  collections.defaultdict(dict)
#     N_plus[1][0] = [ 0, 1 ]
#     N_plus[0][1] = [ 1, 0 ]

#     V = collections.defaultdict(dict)
#     V[1][0] = [ 0, 1 ]
#     V[0][1] = [ 0, 1 ]

#     v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

#     return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,  LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )


# def calculate_signal_matrices(FeedbackMatrix, N,M,A):
#     signal_matrices = []

#     if N == 3: #label efficient
#         for i in range(N):
#             signalMatrix = np.zeros( (A, M) )
#             for j in range(M):
#                 a = FeedbackMatrix[i][j]
#                 signalMatrix[ feedback_idx_label_efficient(a) ][j] = 1
#             signal_matrices.append(signalMatrix)
#     elif N == 2: #apple tasting
#         for i in range(N):
#             signalMatrix = np.zeros( (A,M) )
#             for j in range(M):
#                 a = FeedbackMatrix[i][j]
#                 signalMatrix[ feedback_idx_apple_tasting(a) ][j] = 1
#             signal_matrices.append(signalMatrix)
#     else: #dynamic pricing
#         for i in range(N):
#             signalMatrix = np.zeros( (A, M) )
#             for j in range(M):
#                 a = FeedbackMatrix[i][j]
#                 signalMatrix[ feedback_idx_dynamic_pricing(a) ][j] = 1
#             signal_matrices.append(signalMatrix)


#     return signal_matrices

# def feedback_idx_apple_tasting(feedback):
#     idx = None
#     if feedback == 0:
#         idx = 0
#     elif feedback == 1:
#         idx = 1
#     return idx

# def feedback_idx_label_efficient(feedback):
#     idx = None
#     if feedback == 1:
#         idx = 0
#     elif feedback == 0.5:
#         idx = 1
#     elif feedback == 0.25:
#         idx = 2
#     return idx
    
