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
    FeedbackMatrix =  np.array([ [0, 0],[1, 2] ])
    signal_matrices =  [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ] ) ]

    FeedbackMatrix_PMDMED =  np.array([ [0, 0],[1, 2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,1],[0,0],[0,0] ] ), np.array( [ [0,0],[0,1],[1,0] ] ) ]

    mathcal_N = [ [0, 1] ] #, [1, 0] 

    v = { 0: {1: [np.array([0]), np.array([-1.,  1.])]} } #, 1: {0: [np.array([0]), np.array([ 1., -1.])]}

    N_plus =  collections.defaultdict(dict)
    N_plus[0][1] = [ 0, 1 ]
    # N_plus[1][0] = [ 0, 1 ]

    V = collections.defaultdict(dict)
    V[0][1] = [ 0,1 ]
    # V[1][0] = [ 0,1 ]

    game = Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED,signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

    return game


# def label_efficient(  ):

#     name = 'LE'
#     LossMatrix = np.array( [ [1, 1],[1, 0],[0, 1] ] )
#     FeedbackMatrix = np.array(  [ [1, 1/2], [1/4, 1/4], [1/4, 1/4] ] )
#     LinkMatrix = np.array( [ [0, 2, 2],[2, -2, -2],[-2, 4, 4] ] )
#     signal_matrices = [ np.array( [ [0,1],[1,0] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

#     bandit_LossMatrix = np.array( [ [1, 1], [1, 0], [0, 1] ] )
#     bandit_FeedbackMatrix =  np.array([ [1,-1], [0,0], [0, 0] ])

#     FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2],[2,2] ])
#     A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
#     signal_matrices_Adim =  [ np.array( [ [1,0],[0,1],[0,0] ] ), np.array( [ [0,0],[0,0],[1,1] ] ), np.array( [ [0,0],[0,0],[1,1] ] ) ]
    
#     mathcal_N = [  [1,2],  [2,1] ] 

#     v = {1: {2: [ np.array([-1.,  1.]), np.array([0]), np.array([0])]}, 2: {1: [np.array([ 1., -1.]), np.array([0.]), np.array([0.])]}}
    
#     N_plus =  collections.defaultdict(dict)
#     N_plus[2][1] = [ 1, 2 ]
#     N_plus[1][2] = [ 1, 2 ]

#     V = collections.defaultdict(dict)
#     V[2][1] = [ 0, 1, 2 ]
#     V[1][2] = [ 0, 1, 2 ]

#     return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,  LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )


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

# def feedback_idx_dynamic_pricing(feedback):
#     idx = None
#     if feedback == 1:
#         idx = 0
#     elif feedback == 2:
#         idx = 1
#     return idx

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
    
    
# def dynamic_pricing( mode ):
#     c = 2
#     LossMatrix = np.array( [ [0,1,2,3,4], [c,0,1,2,3],[c,c,0,1,2],[c,c,c,0,1],[c,c,c,c,0] ] )
#     FeedbackMatrix = np.array( [ [2,2,2,2,2], [1,2,2,2,2], [1,1,2,2,2], [1,1,1,2,2], [1,1,1,1,2] ] )

#     SignalMatrices = [  np.array( [  [1,1,1,1,1] ] ), 
#                         np.array( [ [1,0,0,0,0],[0,1,1,1,1] ] ), 
#                         np.array( [ [1,1,0,0,0],[0,0,1,1,1] ] ),
#                         np.array( [ [1,1,1,0,0],[0,0,0,1,1] ] ),
#                         np.array( [ [1,1,1,1,0],[0,0,0,0,1] ] ) ] 

#     A = geometry_v3.alphabet_size( FeedbackMatrix,  len(FeedbackMatrix),len(FeedbackMatrix[0]) )
#     signal_matrices_Adim = calculate_signal_matrices(FeedbackMatrix, len(FeedbackMatrix),len(FeedbackMatrix[0]), A)

#     mathcal_N = [(0, 1),  (0, 2),  (0, 3),  (0, 4),  (1, 0),  (1, 2),  (1, 3),  (1, 4),  (2, 0),  (2, 1),  
#                  (2, 3),  (2, 4),  (3, 0), (3, 1), (3, 2), (3, 4), (4, 0), (4, 1), (4, 2),  (4, 3)]

#     v = {0: {1: {0: np.array([-0.33333333]), 1: np.array([-1.66666667,  1.33333333])},
#               2: {0: np.array([0]),
#                1: np.array([-0.5,  0.5]),
#                2: np.array([-1.5,  1.5]),
#                3: np.array([0,  0]),
#                4: np.array([0, 0])},
#               3: {0: np.array([0.16666667]),
#                1: np.array([-0.41666667,  0.58333333]),
#                2: np.array([-0.41666667,  0.58333333]),
#                3: np.array([-1.41666667,  1.58333333]),
#                4: np.array([0.08333333, 0.08333333])},
#               4: {0: np.array([0.33333333]),
#                1: np.array([-0.33333333,  0.66666667]),
#                2: np.array([-0.33333333,  0.66666667]),
#                3: np.array([-0.33333333,  0.66666667]),
#                4: np.array([-1.33333333,  1.66666667])}},
#              1: {0: {1: np.array([ 1.66666667, -1.33333333]),
#                0: np.array([0.33333333])},
#               2: {1: np.array([ 1.25, -0.75]), 2: np.array([-1.25,  1.75])},
#               3: {0: np.array([0.33333333]),
#                1: np.array([ 1.16666667, -0.83333333]),
#                2: np.array([-0.33333333,  0.66666667]),
#                3: np.array([-1.33333333,  1.66666667]),
#                4: np.array([0.16666667, 0.16666667])},
#               4: {0: np.array([0.5]),
#                1: np.array([ 1.25, -0.75]),
#                2: np.array([-0.25,  0.75]),
#                3: np.array([-0.25,  0.75]),
#                4: np.array([-1.25,  1.75])}},
#              2: {0: {0: np.array([0]),
#                1: np.array([ 0.5, -0.5]),
#                2: np.array([ 1.5, -1.5]),
#                3: np.array([0, 0]),
#                4: np.array([ 0, 0])},
#               1: {2: np.array([ 1.25, -1.75]), 1: np.array([-1.25,  0.75])},
#               3: {2: np.array([ 1.25, -0.75]), 3: np.array([-1.25,  1.75])},
#               4: {0: np.array([0.33333333]),
#                1: np.array([0.16666667, 0.16666667]),
#                2: np.array([ 1.16666667, -0.83333333]),
#                3: np.array([-0.33333333,  0.66666667]),
#                4: np.array([-1.33333333,  1.66666667])}},
#              3: {0: {0: np.array([-0.16666667]),
#                1: np.array([ 0.41666667, -0.58333333]),
#                2: np.array([ 0.41666667, -0.58333333]),
#                3: np.array([ 1.41666667, -1.58333333]),
#                4: np.array([-0.08333333, -0.08333333])},
#               1: {0: np.array([-0.33333333]),
#                1: np.array([-1.16666667,  0.83333333]),
#                2: np.array([ 0.33333333, -0.66666667]),
#                3: np.array([ 1.33333333, -1.66666667]),
#                4: np.array([-0.16666667, -0.16666667])},
#               2: {3: np.array([ 1.25, -1.75]), 2: np.array([-1.25,  0.75])},
#               4: {3: np.array([ 1.25, -0.75]), 4: np.array([-1.25,  1.75])}},
#              4: {0: {0: np.array([-0.33333333]),
#                1: np.array([ 0.33333333, -0.66666667]),
#                2: np.array([ 0.33333333, -0.66666667]),
#                3: np.array([ 0.33333333, -0.66666667]),
#                4: np.array([ 1.33333333, -1.66666667])},
#               1: {0: np.array([-0.5]),
#                1: np.array([-1.25,  0.75]),
#                2: np.array([ 0.25, -0.75]),
#                3: np.array([ 0.25, -0.75]),
#                4: np.array([ 1.25, -1.75])},
#               2: {0: np.array([-0.33333333]),
#                1: np.array([-0.16666667, -0.16666667]),
#                2: np.array([-1.16666667,  0.83333333]),
#                3: np.array([ 0.33333333, -0.66666667]),
#                4: np.array([ 1.33333333, -1.66666667])},
#               3: {4: np.array([ 1.25, -1.75]), 3: np.array([-1.25,  0.75])}}}

#     N_plus =  collections.defaultdict(dict)
#     N_plus[0][1] = [ 0, 1 ]
#     N_plus[0][2] = [ 0, 2 ]
#     N_plus[0][3] = [ 0, 3 ]
#     N_plus[0][4] = [ 0, 4 ]
#     N_plus[1][0] = [ 1, 0 ]
#     N_plus[1][2] = [ 1, 2 ]
#     N_plus[1][3] = [ 1, 3 ]
#     N_plus[1][4] = [ 1, 4 ]
#     N_plus[2][0] = [ 2, 0 ]
#     N_plus[2][1] = [ 2, 1 ]
#     N_plus[2][3] = [ 2, 3 ]
#     N_plus[2][4] = [ 2, 4 ]
#     N_plus[3][0] = [ 3, 0 ]
#     N_plus[3][1] = [ 3, 1 ]
#     N_plus[3][2] = [ 3, 2 ]
#     N_plus[3][4] = [ 3, 4 ]
#     N_plus[4][0] = [ 4, 0 ]
#     N_plus[4][1] = [ 4, 1 ]
#     N_plus[4][2] = [ 4, 2 ]
#     N_plus[4][3] = [ 4, 3 ]

#     V = collections.defaultdict(dict)
#     V[0][1] = [ 0, 1 ]
#     V[0][2] = [0, 1, 2 ,3 ,4]
#     V[0][3] = [0, 1, 2 ,3 ,4]
#     V[0][4] = [0, 1, 2 ,3 ,4]
#     V[1][0] = [ 1, 0 ]
#     V[1][2] = [ 1, 2 ]
#     V[1][3] = [0, 1, 2 ,3 ,4]
#     V[1][4] = [0, 1, 2 ,3 ,4]
#     V[2][0] = [0, 1, 2 ,3 ,4]
#     V[2][1] = [ 2, 1 ]
#     V[2][3] = [ 2, 3 ]
#     V[2][4] = [0, 1, 2 ,3 ,4]
#     V[3][0] = [0, 1, 2 ,3 ,4]
#     V[3][1] = [0, 1, 2 ,3 ,4]
#     V[3][2] = [ 3, 2 ]
#     V[3][4] = [ 3, 4 ]
#     V[4][0] = [0, 1, 2 ,3 ,4]
#     V[4][1] = [0, 1, 2 ,3 ,4]
#     V[4][2] = [0, 1, 2 ,3 ,4]
#     V[4][3] = [ 4, 3 ]

#     LinkMatrix = np.linalg.inv( FeedbackMatrix ) @ LossMatrix 

#     game = Game( LossMatrix, FeedbackMatrix, LinkMatrix, SignalMatrices, signal_matrices_Adim, mathcal_N, v, N_plus, V , mode )

#     return game


# def general_algorithm(F, L):
#     # print(F)

#     N, M = F.shape

#     Fdash_list = []
#     Ldash_list = []
#     # We use the lists sizes for z in paper
#     h = {} # pseudo-action to action map
#     s = {} # pseudo-action to symbol map

#     for j in range(N):

#         for v in set(F[j,...]):
                
#             Fiv = geometry.signal_vec( j, v, F )

#             if not Fdash_list or not geometry.is_linear_comb(Fiv, Fdash_list):
#                 h[len(Fdash_list)] = j # h(z)=i in FeedExp3 paper
#                 s[len(Fdash_list)] = v # not in FeedExp3 paper ??
#                 Fdash_list.append(Fiv)
#                 Ldash_list.append(L[j,...])
#                 bool_fiv_added = True

#         if not bool_fiv_added:
#             h[len(Fdash_list)] = j # h(z)=j in the paper
#             s[len(Fdash_list)] = v # not in FeedExp3 paper ??
#             Fdash_list.append( np.zeros(M) )
#             Ldash_list.append(L[j,...])

#     # Build F' and H' matrices
#     FdashMatrix = np.vstack(Fdash_list)
#     LdashMatrix = np.vstack(Ldash_list)

#     Ndash, Mdash = FdashMatrix.shape
#     assert FdashMatrix.shape == LdashMatrix.shape # just in case

#     # Search for strictly-dominating pseudo-actions 
#     NonEmptyCells = []
#     EmptyCells = []
#     for iv in range(Ndash):
#         if geometry.isStrictlyNonDominated(iv, LdashMatrix):
#             NonEmptyCells.append(iv)
#         else:
#             EmptyCells.append(iv)
            
#     if len(NonEmptyCells)>0: # An empty nonEmptyCells is a problem!
#         # Pick one non-dominated action
#         b = np.random.choice(NonEmptyCells)      # Choose any action from the set of actions with nonempty cells
#     else:
#         print("WARNING: no strictly dominant cell found")
#         #b = random.choice(range(Ndash))  # Choose any action
#         b = 0

#     # Translate the loss relatively to pseudo-action b. Recall that loss transposition does not impact the policy regret.
#     LdashMatrix = LdashMatrix - LdashMatrix[b,...]
    
#     # Makes the dominated actions as bad as possible i.e. with worst possible loss
#     for iv in EmptyCells:
#         if not geometry.isNonDominated(iv, LdashMatrix):
#             LdashMatrix[iv,...] =  max(LdashMatrix[iv,...])
        
#     return FdashMatrix, LdashMatrix