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
import pickle
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR


class DeployedNetwork(nn.Module):
    def __init__(self,  d, m):
        super(DeployedNetwork, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.activate1 = nn.Tanh() #nn.ReLU()
        self.fc2 = nn.Linear(m, m)
        self.activate2 = nn.Tanh() #nn.ReLU()
        self.fc3 = nn.Linear(m, m)
        self.activate3 = nn.Tanh() #nn.ReLU()
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    def forward(self, x):
        x = self.activate3( self.fc3( self.activate2( self.fc2( self.activate1( self.fc1( x ) ) ) ) ) )
        # x = self.fc3( self.activate2( self.fc2( self.activate1( self.fc1( x ) ) ) ) ) 
        # x = self.fc2( self.activate1( self.fc1(x) ) ) 

        # Add an extra element with the value 1
        ones = torch.ones(x.shape[0], 1, device=x.device)
        x = torch.cat((ones, x), dim=1)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, ):
        self.obs = None
        self.feedbacks = None
        self.actions = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.feedbacks[index]
    
    def append(self, X , f):
        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0) 
        self.feedbacks = [[f]] if self.feedbacks is None else np.concatenate( (self.feedbacks, [[f]] ), axis=0)

class CBPside():

    def __init__(self, game, alpha, lbd_neural, lbd_reg, sigma, K, epsilon, m, H, device):

        self.name = 'randneuralcbp'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.SignalMatrices = game.SignalMatrices

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = self.getConfidenceWidth( )
        #print('W', self.W)
        self.alpha = alpha
            
        self.lbd_neural = lbd_neural
        self.lbd_reg = lbd_reg

        self.eta =  self.W**(2/3) 
        self.m = m
        self.H = H
        
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
        self.H = 50
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.func = DeployedNetwork( self.d , self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
        self.feedbacks = []

        self.contexts = {'feats':None, 'r_feats':None, 'labels':None, 'weights': None,
                                    'V_it_inv': self.lbd_reg * np.identity(self.m+1),
                                    'V_i0_inv': self.lbd_reg * np.identity(self.m+1) } 
            
    def obtain_probability(self,  t , factor):

        def divide_interval(start, end, k):
            intervals = np.linspace(start, end, k).tolist()
            return intervals
    
        U =  factor
        rhos = divide_interval(0, U, self.K)
        p_m_hat =  np.array([ np.exp( -(rhos[i]**2) / 2*(self.sigma**2)  )  for i in range(len(rhos)-1) ] )

        p_m = (1 - self.epsilon) * p_m_hat / p_m_hat.sum()
        p_m = p_m.tolist()
        p_m.append(self.epsilon)
            
        Z = np.random.choice(rhos, p= p_m)

        return Z

    def get_action(self, t, X):
        self.X = X
        self.latent_X = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach().numpy()
        # self.latent_X = np.concatenate( [ [[1]], self.latent_X], 1)
        # print('latent', self.latent_X.shape, self.latent_X)

        if t < self.N:
            action = t
            history = {'monitor_action':action, 'explore':1, }
            
        else: 

            halfspace = []
            q = []
            w = []

            for i in range(self.N):

                if i == 0:
                    pred = self.latent_X @ self.contexts['weights'].T
                    pred = pred[0]
                    pred = np.array([1-pred,pred])
                else:
                    pred = np.array([[1]])
                # print('action', i, pred.shape)
                q.append(  pred  )

                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt( 2 * ( self.d  * np.log( 1 + t * np.log(self.N * self.H)/self.lbd_reg ) +  np.log(1/t**2) ) ) + np.sqrt(self.lbd_reg) * sigma_i )
                factor = self.obtain_probability(t, factor)
                width = np.sqrt( self.latent_X @ self.contexts['V_it_inv'] @ self.latent_X.T )
                formule = factor * width
                # print('factor', factor, 'width', width)
                w.append( formule )

            # print()    
            # print( 'estimate', q )
            # print('conf   ', w )

            for pair in self.mathcal_N:
                tdelta, c = 0, 0

                for k in  self.V[ pair[0] ][ pair[1] ]:
                    # print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
                # print('pair', pair, 'tdelta', tdelta, 'confidence', c)
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

            for k,eta in zip(V_t, self.eta):
                if eta>0:
                    val =  self.latent_X @ self.contexts['V_it_inv'] @ self.latent_X.T 
                    t_prime = t
                    with np.errstate(divide='ignore'): 
                        rate = self.eta[k] * t_prime**(2/3)  * ( self.alpha * np.log(t_prime) )**(1/3)  #* self.N**2 * 4 *  self.d**2
                        print(k, val[0][0], 1/rate)
                        if val[0][0] > 1/rate : 
                            # print('append action ', k)
                            # print('action', k, 'threshold', self.eta[k] * geometry_v3.f(t, self.alpha), 'constant', self.eta[k], 'value', geometry_v3.f(t, self.alpha)  )
                            R_t.append(k)

            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)
            
            explored = 1 if len(union1)==2 else 0
        
            print('union1', union1, 'R', R_t)
            S =  np.union1d(  union1  , R_t )
            S = np.array( S, dtype = int)
            # print('S', S)
            S = np.unique(S)
            # print()
            values = { i:self.W[i]*w[i] for i in S}
            # print('value', values)
            action = max(values, key=values.get)

            history = {'monitor_action':action, 'explore':explored, }
            
        return action, history
    
    def reinitialize_exploration(self,):

        actualized_cs = self.func( torch.tensor( self.contexts['r_feats'] ).to(self.device).float() )
        actualized_cs = actualized_cs.cpu().detach().numpy()
        # print('actualized_cs', actualized_cs.shape)
        V_it_inv = self.contexts['V_i0_inv']
        for c in actualized_cs:
            c = np.array(c).reshape( (1, self.m) )
            V_it_inv = V_it_inv - ( V_it_inv @ c.T @ c @ V_it_inv ) / ( 1 + c @ V_it_inv @ c.T ) 
        self.contexts['V_it_inv'] = V_it_inv
        weights = self.contexts['labels'] @ actualized_cs @ self.contexts['V_it_inv']
        self.contexts['weights'] = weights

    def update(self, action, feedback, outcome, t, X):

        if t>1000:
            self.H = 500 

        if action == 0:
            self.hist.append(X, feedback)
            self.contexts['labels'] =  np.array([feedback]) if self.contexts['labels'] is None else np.concatenate( (self.contexts['labels'], [feedback] ), axis=0)
            self.contexts['feats'] = self.latent_X if self.contexts['feats'] is None else np.concatenate( (self.contexts['feats'], self.latent_X), axis=0)
            self.contexts['r_feats'] = self.X if self.contexts['r_feats'] is None else np.concatenate( (self.contexts['r_feats'], self.X), axis=0)

            # print(self.contexts[0]['labels'].shape, self.contexts[0]['feats'].shape )

            V_it_inv = self.contexts['V_it_inv']
            V_it_inv = V_it_inv - ( V_it_inv @ self.latent_X.T @ self.latent_X @ V_it_inv ) / ( 1 + self.latent_X @ V_it_inv @ self.latent_X.T ) 
            self.contexts['V_it_inv'] = V_it_inv
            weights = self.contexts['labels'] @ self.contexts['feats'] @ self.contexts['V_it_inv']
            self.contexts['weights'] = weights
            
        global_loss = []
        global_losses = []

        if (t>self.N):

            if (t % 50 == 0 and t<1000) or (t % 500 == 0 and t>=1000):

                # self.reinitialize_exploration()

                self.weights = self.contexts['weights'] 
                self.func = copy.deepcopy(self.func0)
                optimizer = optim.Adam(self.func.parameters(), lr=0.1, weight_decay = self.lbd_neural )
                dataloader = DataLoader(self.hist, batch_size=1000, shuffle=True) 
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
                # scheduler = StepLR(optimizer, step_size=250, last_epoch=-1, gamma=0.1)
                for _ in range(1000): 
                    
                    train_loss, losses = self.step(dataloader, optimizer)
                    current_lr = optimizer.param_groups[0]['lr']
                    global_loss.append(train_loss)
                    global_losses.append(losses)

                    if _ % 10 == 0 :
                        scheduler.step()
                    if _ % 25 == 0:
                        print('train loss', train_loss, 'losses', losses )

        return global_loss, global_losses
                

    def step(self, loader, opt):

        for X, feedbacks in loader:
            X = X.to(self.device).float() 
            feedbacks = feedbacks.to(self.device).float()

            losses = []
            losses_vec =[]
    
            weights = torch.from_numpy( self.weights ).unsqueeze(0).float().to(self.device)
            pred = self.func(X) @ weights.T
            l = nn.MSELoss()(pred, feedbacks)

            losses.append( l )
            losses_vec.append( l.item() )

            opt.zero_grad()
            l.backward()
            opt.step()
            # print(losses)
        return l.item(), losses_vec


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
