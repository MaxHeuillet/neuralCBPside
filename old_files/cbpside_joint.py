import numpy as np
import geometry_gurobi

class CBPside():

    def __init__(self, game, alpha, lbd):

        self.name = 'cbpsidejoint'

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_gurobi.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        # print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = game.SignalMatrices
        # print('signalmatrices', self.SignalMatrices)

        self.pareto_actions = geometry_gurobi.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        # print('mathcal_N', self.mathcal_N)

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = self.getConfidenceWidth( )
        #print('W', self.W)
        self.alpha = alpha
        self.lbd = lbd

        self.eta =  self.W **2/3 


    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            # print('pair', pair, 'N_plus', N_plus[ pair[0] ][ pair[1] ] )
            for k in self.V[ pair[0] ][ pair[1] ]:
                # print('pair ', pair, 'v ', v[ pair[0] ][ pair[1] ], 'V ', V[ pair[0] ][ pair[1] ] )
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec , np.inf) ] )
        return W

    def reset(self, d):
        self.d = d
        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)]  #[ np.zeros(    len( np.unique(self.game.FeedbackMatrix[i] ) )  ) for i in range(self.N)] 
        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.features = None
        self.labels = None
        self.weights = None
        self.V_t_inv = np.identity(self.d * self.A)

 
    def get_action(self, t, X):

        if t < self.N:
            action = t
            tdelta = 0
            history = [t, np.nan, np.nan]
            # self.contexts[t]['weights'] = self.SignalMatrices[t] @ np.array( [ [0,1],[1,-1] ])

        else: 

            pred_buffer = {i: [] for i in range(self.N)}
            halfspace = []
            q = []
            w = []
            
            unique_elements = np.unique(self.game.FeedbackMatrix)
            for signal in unique_elements:
                act_to_idx = np.where(self.game.FeedbackMatrix == signal)[0][0]
                pred = self.weights @ X[signal]
                # print( pred.shape )
                pred_buffer[act_to_idx].append( pred )
            
            for i in range(self.N):

                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd) * sigma_i )
                width = np.sqrt( X[signal].T @ self.V_t_inv @ X[signal] )
                formule = factor * width
                w.append( formule )
                q.append( np.array(pred_buffer[i]).reshape(sigma_i,1) )
            # print()    
            print( 'estimate', q )
            print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta = np.zeros( (1,) )
                c = 0

                for k in  self.V[ pair[0] ][ pair[1] ]:
                    # print( 'pair ', pair, 'action ', k, 'proba ', self.nu[k]  / self.n[k]  )
                    # print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                # print('sign', np.sign(tdelta) )
                tdelta = tdelta[0]
                # c =  np.inf
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta) ) ) 
            
            # print('halfspace', halfspace)
            P_t = self.pareto_halfspace_memory(halfspace)
            N_t = self.neighborhood_halfspace_memory(halfspace)

            Nplus_t = []
            for pair in N_t:
                Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
            Nplus_t = np.unique(Nplus_t)

            V_t = []
            for pair in N_t:
                V_t.extend( self.V[ pair[0] ][ pair[1] ] )
            V_t = np.unique(V_t)

            R_t = []
            # for k in V_t:
            #   val =  X.T @ self.contexts[k]['V_it_inv'] @ X
            #   t_prime = t
            #   with np.errstate(divide='ignore'): 
            #     rate = np.sqrt( self.eta[k] * self.N**2 * 4 *  self.d**2  *(t_prime**(2/3) ) * ( self.alpha * np.log(t_prime) )**(1/3) ) 
            #     # print(k, val[0][0], 1/rate)
            #     if val[0][0] > 1/rate : 
            #         # print('append action ', k)
            #         # print('action', k, 'threshold', self.eta[k] * geometry_v3.f(t, self.alpha), 'constant', self.eta[k], 'value', geometry_v3.f(t, self.alpha)  )
            #         R_t.append(k)

            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)
            print('union1', union1)
            S =  np.union1d(  union1  , R_t )
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
            # print('outcome frequency', self.nu, 'action frequency', self.n )
            #print()
            values = { i:self.W[i]*w[i] for i in S}
            # print('value', values)
            action = max(values, key=values.get)
            # print('S',S, 'values', values, 'action', action)
            # 'P_t',P_t,'N_t', N_t,'Nplus_t',Nplus_t,'V_t',V_t, 'R_t',R_t,  print('n', self.n,'nu', self.nu)
            # print()

            history = [ action, factor, tdelta ]

        return action, history

    def update(self, action, feedback, outcome, t, X):

        e_y = np.zeros( (self.M,1) )
        e_y[outcome] = 1
        Y_t = self.game.SignalMatricesAdim[action] @ e_y 

        self.features = X if self.features is None else np.concatenate((self.features, X), axis=0)
        self.labels = Y_t.T if self.labels is None else np.concatenate((self.labels, Y_t.T), axis=1)

        # print('X', X.shape, X[0].shape, np.expand_dims(X[0], axis=1).shape )
        
        # # print('Yit shape',Y_it.shape)
        # print( 'features', self.features.shape )
        # print( 'labels', self.labels.shape )    
        # print( 'V_t_inv', self.V_t_inv.shape )    

        for i in range(self.A):
            Xi = np.expand_dims(X[i], axis=1)
            V_t_inv = self.V_t_inv
            self.V_t_inv = V_t_inv - ( V_t_inv @ Xi @ Xi.T @ V_t_inv ) / ( 1 + Xi.T @ V_t_inv @ Xi ) 
        weights = self.labels @ self.features @ self.V_t_inv
        self.weights = weights
        # print( 'weigts', weights )
        # print()
        # print('action', action, 'Y_t', Y_t, 'shape', Y_t.shape, 'nu[action]', self.nu[action], 'shape', self.nu[action].shape)


    def halfspace_code(self, halfspace):
        string = ''
        for element in halfspace:
            pair, sign = element
            string += '{}{}{}'.format(pair[0],pair[1], sign)
        return string 


    def pareto_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_pareto.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_pareto[ code ]
        else:
            result =  geometry_gurobi.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            self.memory_pareto[code ] =result
 
        return result

    def neighborhood_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_neighbors.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_neighbors[ code ]
        else:
            result =  geometry_gurobi.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )
            self.memory_neighbors[code ] =result
 
        return result
