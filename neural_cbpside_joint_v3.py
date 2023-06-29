import numpy as np
import geometry_v3


import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize
import copy

class Network(nn.Module):
    def __init__(self,  d, m):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(m, m)
        self.activate2 = nn.ReLU()
    def forward(self, x):
        x = self.activate2( self.fc2( self.activate1( self.fc1(x) ) ) )
        return x
    
def convert_list(A):
    B = []
    B.append(np.array([A[0]]).reshape(1, 1))
    sub_array = np.array(A[1:]).reshape(2, 1)
    B.append(sub_array)
    return B


from torch.utils.data import Dataset

class ExperienceReplay(Dataset):
    def __init__(self, size):
        self.size = size
        self.obs = None
        self.latent_obs = None
        self.labels = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.latent_obs[index], self.labels[index]

    def append(self, X, X_latent, y):
        if self.obs is not None and len(self.obs) >= self.size :
            self.obs = self.obs[1:, :]
            self.latent_obs = self.latent_obs[1:, :]
            self.labels = self.labels[1:, :]

        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0)
        self.latent_obs = X_latent if self.latent_obs is None else np.concatenate( (self.latent_obs, X_latent), axis=0) 
        self.labels = y if self.labels is None else np.concatenate( (self.labels, y), axis=0)

    def clear(self):
        self.obs = None
        self.latent_obs = None
        self.labels = None



class CBPside():

    def __init__(self, game, alpha, lbd_neural, lbd_reg, m, device):

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
            
        self.lbd_neural = lbd_neural
        self.lbd_reg = lbd_reg

        self.eta =  self.W ** 2/3 
        self.m = m
        self.H = 50


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
        self.weights = None
        self.A_t_inv = self.lbd_reg * np.identity(self.m)
        self.func = Network( self.d * self.A, self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.replay = ExperienceReplay(128)
        

 
    def get_action(self, t, X):

        self.latent_X = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach().numpy()
        # self.latent_X = X

        # print('latent X', self.latent_X.shape )

        if t < self.N:
            action = t
            history = [t, np.nan, np.nan]
            
        else: 

            halfspace = []
            q = []
            w = []

            pred = self.latent_X @ self.weights.T 
            q = convert_list( pred.squeeze() )

            for i in range(self.N):
                sigma_i = len(self.SignalMatrices[i])
                # factor = sigma_i * (  np.sqrt(  self.d * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd_reg) * sigma_i )
                factor = sigma_i * (  np.sqrt( 2 * ( self.d  * np.log( 1 + t * np.log(self.N * 1)/self.lbd_reg ) +  np.log(1/t**2) ) ) + np.sqrt(self.lbd_reg) * sigma_i )
                width = np.sqrt( self.latent_X[i].T @ self.A_t_inv @ self.latent_X[i] )
                formule = factor * width
                print('factor', factor, 'width', width)
                w.append( formule )

            # print()    
            print( 'estimate', q )
            print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta, c = 0, 0

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

        self.replay.append( X, self.latent_X , Y_t )

        for i in range(self.A):
            Xi = np.expand_dims(self.latent_X[i], axis=1)
            A_t_inv = self.A_t_inv
            self.A_t_inv = A_t_inv - ( A_t_inv @ Xi @ Xi.T @ A_t_inv ) / ( 1 + Xi.T @ A_t_inv @ Xi ) 
        
        # print(t, self.replay.labels.shape, self.replay.latent_obs.shape, self.A_t_inv.shape)
        self.weights = self.replay.labels.T @ self.replay.latent_obs @ self.A_t_inv

        # print('replay shape', self.replay.obs.shape, self.replay.latent_obs.shape, self.replay.labels.shape,)
        optimizer = optim.SGD(self.func.parameters(), lr=10e-5, weight_decay=self.lbd_neural) 
        dataloader = DataLoader(self.replay, batch_size=self.replay.size, shuffle=True) 

        for _ in range(1):
            train_loss = self.SGD_step(dataloader, optimizer)



    def SGD_step(self, loader, opt=None):
        #""Standard training/evaluation epoch over the dataset"""
        lin_weights = torch.tensor(self.weights).float().to(self.device)

        for X, _, y in loader:
            # print('X and y', X.shape, y.shape )
            X,y = X.to(self.device).float(), y.to(self.device).float() 
            
            # print('latent prediction', self.func(X).shape )
            pred = self.func(X) @ lin_weights.T
            # print('pred', pred, pred.shape)
            loss = nn.MSELoss()(pred, y)

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
