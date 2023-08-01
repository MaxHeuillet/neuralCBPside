import numpy as np
import geometry_v3

class CBPside():

    def __init__(self, game, alpha, lbd, sigma, K , epsilon):

        self.name = 'cbpside'

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
        # print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

        self.SignalMatrices = game.SignalMatrices
        # print('signalmatrices', self.SignalMatrices)

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

        self.eta =  self.W**(2/3) 

        self.sigma = sigma
        self.K = K
        self.epsilon = epsilon


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
        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':None, 'labels':None, 'weights': None, 
                                   'V_it_inv': self.lbd * np.identity(self.d) } )
            
    def obtain_probability(self,  t, factor):
    
        U = factor
        rhos = np.arange(0, U, U/self.K )
        p_m_hat =  np.array([ np.exp( -(rhos[i]**2) / 2*(self.sigma**2)  )  for i in range(len(rhos)-1) ] )

        p_m = (1 - self.epsilon) * p_m_hat / p_m_hat.sum()
        p_m = p_m.tolist()
        p_m.append(self.epsilon)
            
        Z = np.random.choice(rhos, p= p_m)

        return Z
 
    def get_action(self, t, X , mode = 'train'):

        # print('X', X.shape)

        if t < self.N:
            action = t
            history = [t, np.nan, np.nan]
            # self.contexts[t]['weights'] = self.SignalMatrices[t] @ np.array( [ [0,1],[1,-1] ])

        else: 

            halfspace = []
            q = []
            w = []
            
            for i in range(self.N):
                # # print( self.contexts[i]['weights'] )
                # print('context shape', X.shape)
                # print('weights shape', self.contexts[i]['weights'].shape)
                pred = X @ self.contexts[i]['weights'].T
                print('action', i, pred.shape)
                q.append(  pred  )

                # factor = self.d * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + len(self.SignalMatrices[i]) )
                # factor = self.d * (  np.sqrt(  (self.d+1) * np.log(t)  ) + len(self.SignalMatrices[i]) )
                # factor = 1
                # factor =  sigma_i * (  np.sqrt(  (self.d+1) * np.log(t)  ) +  sigma_i )
                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd) * sigma_i )
                Z = self.obtain_probability(t, factor)
                width = np.sqrt( X @ self.contexts[i]['V_it_inv'] @ X.T )
                formule = Z * width
                # print('factor', factor, 'width', width)
                # b = X.T @ np.linalg.inv( self.lbd * np.identity(D) + X_it @ X_it.T  ) @ X 
                #print('action {}, first component {}, second component, {}'.format(i, a, b  ) )
                #print('Xit', X_it.shape  )
                w.append( formule )
            # print()    
            print( 'estimate', q )
            print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta , c = 0 , 0

                # print( self.v[ pair[0] ][ pair[1] ][0].shape )
                # print( self.v[ pair[0] ][ pair[1] ][1].shape )

                # print('pair', pair, 'N_plus', self.N_plus[ pair[0] ][ pair[1] ] )
                for k in  self.V[ pair[0] ][ pair[1] ]:
                    # print( 'pair ', pair, 'action ', k, 'proba ', self.nu[k]  / self.n[k]  )
                    # print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k].T
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                # print('sign', np.sign(tdelta) )
                tdelta = tdelta[0]
                # c =  np.inf
                if mode == 'eval':
                    c = 0
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
              val =  X @ self.contexts[k]['V_it_inv'] @ X.T
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
            history = [ action, factor, tdelta ]

        return action, history

    def update(self, action, feedback, outcome, t, X):

        e_y = np.zeros( (self.M, 1) )
        e_y[outcome] = 1 
        Y_t =  self.game.SignalMatrices[action] @ e_y 


        print(Y_t.shape, X.shape)
        self.contexts[action]['labels'] = Y_t if self.contexts[action]['labels'] is None else np.concatenate( (self.contexts[action]['labels'], Y_t), axis=1)
        self.contexts[action]['features'] = X if self.contexts[action]['features'] is None else np.concatenate( (self.contexts[action]['features'], X), axis=0)

        print(self.contexts[action]['labels'].shape, self.contexts[action]['features'].shape)
        
        V_it_inv = self.contexts[action]['V_it_inv']
        self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X.T @ X @ V_it_inv ) / ( 1 + X @ V_it_inv @ X.T ) 
        weights = self.contexts[action]['labels'] @ self.contexts[action]['features'] @ self.contexts[action]['V_it_inv']
        self.contexts[action]['weights'] = weights



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
