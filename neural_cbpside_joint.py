import numpy as np
import geometry_v3


import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.optimize import minimize
import copy

class Network(nn.Module):
    def __init__(self, output_dim, dim, hidden_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
    def forward(self, x):
        x = self.activate( self.fc1(x) )
        return x

class CBPside():

    def __init__(self, game, alpha, lbd, m, device):

        self.name = 'cbpsidejoint'
        self.device = device

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

        self.eta =  self.W ** 2/3 
        self.m = m


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
        self.A_t_inv = np.identity(self.m)
        self.b_t = np.zeros( (self.m,1) )
        self.func = Network( 1, self.d * self.A, hidden_size=self.m).to(self.device)
        self.latent_buffer = None
        self.latent_features = None
        

 
    def get_action(self, t, X):

        if t < self.N:
            action = t
            tdelta = 0
            history = [t, np.nan, np.nan]
            self.latent_buffer = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach().numpy()
            # self.contexts[t]['weights'] = self.SignalMatrices[t] @ np.array( [ [0,1],[1,-1] ])

        else: 

            pred_buffer = {i: [] for i in range(self.N)}
            halfspace = []
            q = []
            w = []
            
            unique_elements = [0, 2, 1] #np.unique(self.game.FeedbackMatrix)
            for signal in unique_elements:
                # print(  'X dim', torch.from_numpy( X[signal] ).shape )
                act_to_idx = np.where(self.game.FeedbackMatrix == signal)[0][0]
                # print('weights', self.weights, self.weights.shape)
                # print('pred', self.func( X[signal] ), self.func( X[signal] ).shape)
                latent_rep =   self.func( torch.from_numpy( X[signal] ).float().to(self.device) ).cpu().detach().numpy()
                # print('latent_rep and weights', self.weights.shape, latent_rep.shape,)
                pred = self.weights.T @ latent_rep
                # print('pred', pred)
                # print('final', pred, pred.shape )
                pred_buffer[act_to_idx].append( pred )
                self.latent_buffer = latent_rep if self.latent_buffer is None else np.vstack((self.latent_buffer, latent_rep) )
            # print('latent buffer', self.latent_buffer.shape)

            for i in range(self.N):
                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt(  self.m * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd) * sigma_i )
                width = np.sqrt( self.latent_buffer[signal].T @ self.A_t_inv @ self.latent_buffer[signal] )
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
        # print('Y_t', Y_t.shape)

        self.labels = Y_t if self.labels is None else np.vstack( (self.labels, Y_t) )

        self.features = X if self.features is None else np.concatenate( (self.features, X), axis=0)

        self.latent_features = self.latent_buffer if self.latent_features is None else np.vstack( (self.latent_features, self.latent_buffer) )

        # print('X', X.shape, X[0].shape, np.expand_dims(X[0], axis=1).shape )
        # # print('Yit shape',Y_it.shape)
        # print('latent buffer', self.latent_buffer.shape )
        # print( 'features', self.features.shape )
        # print( 'labels', self.labels.shape )    
        # print( 'V_t_inv', self.V_t_inv.shape )    

        # update linear model:

        # for i in range(self.A):
        #     Xi = np.expand_dims(X[i], axis=1)
        #     V_t_inv = self.V_t_inv
        #     self.V_t_inv = V_t_inv - ( V_t_inv @ Xi @ Xi.T @ V_t_inv ) / ( 1 + Xi.T @ V_t_inv @ Xi ) 

        # print('latent buffer', self.latent_buffer.shape)
        for i in range(self.A):
            Xi = np.expand_dims(self.latent_buffer[i], axis=1)
            # print('Xi',Xi)
            # print('Yi', np.squeeze(Y_t[i], 0) )
            A_t_inv = self.A_t_inv
            self.A_t_inv = A_t_inv - ( A_t_inv @ Xi @ Xi.T @ A_t_inv ) / ( 1 + Xi.T @ A_t_inv @ Xi ) 
            self.b_t +=  Y_t[i] * Xi

        # weights = self.labels @ self.latent_features @ self.V_t_inv
        # print('weight shapes', self.A_t_inv.shape, self.b_t.shape)
        weights = self.A_t_inv @ self.b_t
        self.weights = weights
        # print('weights', self.weights)
        if t>self.N:

            # update non-linear model :

            optimizer = optim.SGD(self.func.parameters(), lr=0.01, weight_decay=self.lbd) 
            length = self.labels.shape[0]
            X_tensor = torch.tensor(self.features)
            y_tensor = torch.tensor(self.labels)
            # print('X and y shape', X_tensor.shape, y_tensor.shape)
            dataset = TensorDataset(X_tensor, y_tensor)
            # print('dataset',dataset[0])

            if length < 1000:
                dataloader = DataLoader(dataset, batch_size=length, shuffle=True)
            else:
                dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

            train_loss = self.epoch(dataloader, optimizer)
        self.latent_buffer = None

        # print( 'weigts', weights )
        # print()
        # print('action', action, 'Y_t', Y_t, 'shape', Y_t.shape, 'nu[action]', self.nu[action], 'shape', self.nu[action].shape)


    def epoch(self, loader, opt=None):
        #""Standard training/evaluation epoch over the dataset"""
        expected_dtype = next(self.func.parameters()).dtype
        lin_weights = torch.tensor(self.weights).float().to(self.device)
        # print( 'batch size' , loader.batch_size )
        # lin_weights_matrix = lin_weights.expand( lin_weights.shape[0], loader.batch_size)
        # print('lin weights ', lin_weights.shape)
        for X,y in loader:
            X = X.to(self.device).to(dtype=expected_dtype)
            y = y.to(self.device).to(dtype=expected_dtype)
            # print('X and y', X.shape, y.shape )
            # print('latent prediction', self.func(X).shape )
            pred = self.func(X) @ lin_weights 
            # print('pred', pred, pred.shape)
            loss = nn.MSELoss()(pred, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
        return loss.item()


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
