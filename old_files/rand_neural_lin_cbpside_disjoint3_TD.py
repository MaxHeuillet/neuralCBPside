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
        return x
    
class CustomDataset(Dataset):
    def __init__(self, ):
        self.obs = None
        self.labels = None
        self.feedbacks = None
        self.actions = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.labels[index], self.feedbacks[index], self.actions[index]
    
    def append(self, X , y, f, a):
        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0) 
        self.labels = y if self.labels is None else np.concatenate( (self.labels, y), axis=0)
        self.feedbacks = [[f]] if self.feedbacks is None else np.concatenate( (self.feedbacks, [[f]] ), axis=0)
        self.actions = [[a]] if self.actions is None else np.concatenate( (self.actions, [[a]] ), axis=0)

class CBPside():

    def __init__(self, game, budget, alpha, info_actions, lbd_neural, lbd_reg, sigma, K, epsilon, m, H, device):

        self.name = 'neurallinrandcbpsidedisjoint'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.SignalMatrices = game.SignalMatrices

        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        self.informative_actions = info_actions

        self.budget = budget
        self.counter = 0
        self.over_budget = False

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
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.func = DeployedNetwork( self.d , self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
        self.feedbacks = []

        self.over_budget = False
        self.counter = 0

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'feats':None, 'labels':None, 'weights': None,
                                    'V_it_inv': self.lbd_reg * np.identity(self.m),
                                    'V_i0_inv': self.lbd_reg * np.identity(self.m) } )
            
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

    def get_action(self, t, X, y_pred):

        self.latent_X = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach().numpy()
        # print('latent', self.latent_X.shape, self.latent_X)

        if t < self.N:
            action = t
            history = {'monitor_action':action, 'explore':1, 'model_pred':y_pred, 'counter':self.counter, 'over_budget':self.over_budget}
            
        else: 

            halfspace = []
            q = []
            w = []

            for i in range(self.N):

                if i in self.informative_actions:
                    pred = self.latent_X @ self.contexts[i]['weights'].T
                    pred = pred[0]
                    pred = np.array([pred,1-pred])
                else:
                    pred = np.array([[1]])
                # print('action', i, pred.shape)
                q.append(  pred  )

                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt( 2 * ( self.d  * np.log( 1 + t * np.log(self.N * self.H)/self.lbd_reg ) +  np.log(1/t**2) ) ) + np.sqrt(self.lbd_reg) * sigma_i )
                factor = self.obtain_probability(t, factor)
                width = np.sqrt( self.latent_X @ self.contexts[i]['V_it_inv'] @ self.latent_X.T )
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
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k].T
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
                # print('sign', np.sign(tdelta) )
                tdelta = tdelta[0]
                if self.over_budget:
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
                val =  self.latent_X @ self.contexts[k]['V_it_inv'] @ self.latent_X.T 
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

            history = {'monitor_action':action, 'explore':explored, 'model_pred':y_pred, 'counter':self.counter, 'over_budget':self.over_budget}
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if self.counter > self.budget:
            self.over_budget = True

        if action == 0:
            self.counter += 1

        ### update exploration component:
        e_y = np.zeros( (self.M,1) )
        e_y[outcome] = 1
        Y_t = self.game.SignalMatrices[action] @ e_y 
        print('Y_t', Y_t)

        # print('action', action, 'feedback', feedback, 'Y_t', Y_t, 'latentX', self.latent_X)

        self.contexts[action]['labels'] = Y_t if self.contexts[action]['labels'] is None else np.concatenate( (self.contexts[action]['labels'], Y_t), axis=1)
        self.contexts[action]['feats'] = self.latent_X if self.contexts[action]['feats'] is None else np.concatenate( (self.contexts[action]['feats'], self.latent_X), axis=0)

        V_it_inv = self.contexts[action]['V_it_inv']
        V_it_inv = V_it_inv - ( V_it_inv @ self.latent_X.T @ self.latent_X @ V_it_inv ) / ( 1 + self.latent_X @ V_it_inv @ self.latent_X.T ) 
        self.contexts[action]['V_it_inv'] = V_it_inv
        weights = self.contexts[action]['labels'] @ self.contexts[action]['feats'] @ self.contexts[action]['V_it_inv']
        self.contexts[action]['weights'] = weights

        # print('weights', weights.shape, 'Y_t', Y_t.shape, )
        self.hist.append(X, Y_t, feedback, action)
        global_loss = []
        global_losses = []
        if (t>self.N) and (t % self.H == 0):  

            self.weights = np.vstack( [ self.contexts[i]['weights'] for i in self.informative_actions ] )
            self.func = copy.deepcopy(self.func0)
            optimizer = optim.Adam(self.func.parameters(), lr=0.1, weight_decay = self.lbd_neural )
            dataloader = DataLoader(self.hist, batch_size=len(self.hist), shuffle=True) 
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
            # scheduler = StepLR(optimizer, step_size=250, last_epoch=-1, gamma=0.1)
            for _ in range(1000): 
                
                train_loss, losses = self.step(dataloader, optimizer)
                current_lr = optimizer.param_groups[0]['lr']
                global_loss.append(train_loss)
                global_losses.append(losses)

                if _ % 10 == 0 :
                    scheduler.step()
                # scheduler.step()
                if _ % 25 == 0:
                    print('train loss', train_loss, 'losses', losses )

        return global_loss, global_losses
                

    def step(self, loader, opt):
        #""Standard training/evaluation epoch over the dataset"""

        # symbols = [ np.unique(self.game.FeedbackMatrix[i,...]) for i in range(self.N) ]
        symbols = [ [0] ]

        for X, y, feedbacks, actions in loader:
            X, y  = X.to(self.device).float(), y.to(self.device).float()
            fdks = torch.nn.functional.one_hot(feedbacks[:,0], num_classes= self.A).to(self.device).float()
            loss = 0
            losses = []
            losses_vec =[]
            for i in self.informative_actions:  
                mask = (actions == i)[:,0]
                X_filtered = X[mask]
                fdks_filtered = fdks[mask]
                for s in symbols[i]:
                    y_filtered = fdks_filtered[:,s].unsqueeze(1)
                    weights = torch.from_numpy( self.weights[s] ).unsqueeze(0).float().to(self.device)
                    pred = self.func(X_filtered) @ weights.T
                    # print('pred',pred.shape,'y_filtered', y_filtered.shape)
                    l = nn.MSELoss()(pred, y_filtered)
                    loss += l
                    losses.append( l )
                    losses_vec.append(l.item())
            # Stack the loss elements into a tensor
            # print('losses before', losses)
            loss_tensor = torch.stack(losses)
            # print('losses after', loss_tensor)
            loss_sum = torch.sum(loss_tensor)
            # print('losses sum', loss_sum)
            # ch.tensor(losses).to(self.device)
            # print(loss_sum )
            opt.zero_grad()
            loss_sum.backward()
            opt.step()
            # print(losses)
        return loss.item(), losses_vec


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