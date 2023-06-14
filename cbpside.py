import numpy as np
import geometry_v3

class CBPside():

    def __init__(self, game, d, alpha, lbd):

        self.game = game
        self.d = d

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        # print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = game.SignalMatrices
        # print('signalmatrices', self.SignalMatrices)

        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)]  #[ np.zeros(   ( len( set(game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)] 
        # print('nu', self.nu)

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
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

        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[], 'weights': None, 'V_it_inv': np.identity(self.d) } )


    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            # print('pair', pair, 'N_plus', N_plus[ pair[0] ][ pair[1] ] )
            for k in self.V[ pair[0] ][ pair[1] ]:
                # print('pair ', pair, 'v ', v[ pair[0] ][ pair[1] ], 'V ', V[ pair[0] ][ pair[1] ] )
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec , np.inf) ] )
        return W

    def reset(self,):
        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)]  #[ np.zeros(    len( np.unique(self.game.FeedbackMatrix[i] ) )  ) for i in range(self.N)] 
        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[], 'weights': None, 'V_it_inv': np.identity(self.d) } )

 
    def get_action(self, t, X):

        if t < self.N:
            action = t
            # self.contexts[t]['weights'] = self.SignalMatrices[t] @ np.array( [ [0,1],[1,-1] ])

        else: 

            halfspace = []
            q = []
            w = []
            
            for i in range(self.N):
                # # print( self.contexts[i]['weights'] )
                # print('context shape', X.shape)
                # print('weights shape', self.contexts[i]['weights'].shape)
                
                q.append( self.contexts[i]['weights'] @ X  )

                # factor = self.d * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + len(self.SignalMatrices[i]) )
                # factor = self.d * (  np.sqrt(  (self.d+1) * np.log(t)  ) + len(self.SignalMatrices[i]) )
                # factor = 1
                # factor =  sigma_i * (  np.sqrt(  (self.d+1) * np.log(t)  ) +  sigma_i )
                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd) * sigma_i )
                width = np.sqrt( X.T @ self.contexts[i]['V_it_inv'] @ X )
                formule = factor * width
                print('factor', factor, 'width', width)
                # b = X.T @ np.linalg.inv( self.lbd * np.identity(D) + X_it @ X_it.T  ) @ X 
                #print('action {}, first component {}, second component, {}'.format(i, a, b  ) )
                #print('Xit', X_it.shape  )
                w.append( formule )
            # print()    
            print( 'estimate', q )
            # print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta = np.zeros( (1,) )
                c = 0

                # print( self.v[ pair[0] ][ pair[1] ][0].shape )
                # print( self.v[ pair[0] ][ pair[1] ][1].shape )

                # print('pair', pair, 'N_plus', self.N_plus[ pair[0] ][ pair[1] ] )
                for k in  self.V[ pair[0] ][ pair[1] ]:
                    # print( 'pair ', pair, 'action ', k, 'proba ', self.nu[k]  / self.n[k]  )
                    # print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                # print('sign', np.sign(tdelta) )
                tdelta = tdelta[0]
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
            for k in V_t:
              val =  X.T @ self.contexts[k]['V_it_inv'] @ X
              t_prime = t
              with np.errstate(divide='ignore'): 
                rate = np.sqrt( self.eta[k] * self.N**2 * 4 *  self.d**2  *(t_prime**(2/3) ) * ( self.alpha * np.log(t_prime) )**(1/3) ) 
                # print(k, val[0][0], 1/rate)
                if val[0][0] > 1/rate : 
                    # print('append action ', k)
                    # print('action', k, 'threshold', self.eta[k] * geometry_v3.f(t, self.alpha), 'constant', self.eta[k], 'value', geometry_v3.f(t, self.alpha)  )
                    R_t.append(k)

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

        return action

    def update(self, action, feedback, outcome, t, X):

        e_y = np.zeros( (self.M, 1) )
        e_y[outcome] = 1 
        Y_t =  self.game.SignalMatrices[action] @ e_y 

        self.contexts[action]['labels'].append( Y_t )
        self.contexts[action]['features'].append( X )
        #print(self.contexts[action]['labels']) 
        
        Y_it = np.array( self.contexts[action]['labels'] )
        X_it =  np.array( self.contexts[action]['features'] )

        # n, d, _ = X_it.shape
        # n, sigma, _ = Y_it.shape
        Y_it =  np.squeeze(Y_it, 2).T # Y_it.reshape( (sigma, n) )
        X_it =  np.squeeze(X_it, 2).T #X_it.reshape( (d, n) )

        # print(X_it.shape)
        
        print('Yit shape',Y_it.shape)
        
        V_it_inv = self.contexts[action]['V_it_inv']
        self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 
        weights = Y_it @ X_it.T @ self.contexts[action]['V_it_inv']
        self.contexts[action]['weights'] = weights
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
            result =  geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
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
            result =  geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )
            self.memory_neighbors[code ] =result
 
        return result
