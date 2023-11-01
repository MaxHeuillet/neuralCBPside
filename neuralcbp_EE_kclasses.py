import numpy as np

import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize
import copy
import pickle
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import multiprocessing
import os

# import geometry_gurobi
import geometry_pulp



def sherman_morrison_update(V_it_inv, feature):
    
    V_feature = torch.matmul(V_it_inv, feature.t())
    feature_V = torch.matmul(feature, V_it_inv)
    denom = 1 + torch.matmul(feature_V, feature.t())
    V_it_inv -= torch.matmul(V_feature, feature_V) / denom

    return V_it_inv

class Network_exploitation(nn.Module):
    def __init__(self, input_dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class Network_exploration(nn.Module):
    def __init__(self, input_dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
def EE_forward(net1, net2, x):
    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.backward()
    dc = torch.cat([x.grad.data.detach(), x.detach()], dim=1)
    dc = dc / torch.linalg.norm(dc)
    f2 = net2(dc)
    return f1.item(), f2.item(), dc.to(torch.float16)
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CBPside():

    def __init__(self, game, alpha, lbd_neural, lbd_reg, m, H, num_cls, device):

        self.name = 'neuralCBPside'
        self.device = device
        
        self.num_workers = 1 #int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
        print('num workers', self.num_workers  )

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = len( np.unique( game.FeedbackMatrix ) )

        self.SignalMatrices = game.SignalMatrices

        self.pareto_actions = geometry_pulp.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [], self.num_workers)
        self.mathcal_N = game.mathcal_N

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = self.getConfidenceWidth( )

        self.alpha = alpha
            
        self.lbd_neural = lbd_neural
        self.lbd_reg = lbd_reg

        self.eta =  self.W**(2/3) 
        self.m = m
        self.H = H

        self.num_cls = num_cls

    def convert_pred_format(self,pred):
        final = []
        for k in range(self.game.N):
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                if np.unique(self.game.FeedbackMatrix[k]).shape[0] > 1:
                    per_action.append( pred[s] )
                else:
                    per_action.append( pred[s] )
            final.append( np.array(per_action) )
        return final

    def convert_conf_format(self,conf, dc_list):
        final = []
        for k in range(self.game.N):
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                if np.unique(self.game.FeedbackMatrix[k]).shape[0] > 1:
                    per_action.append( conf[s] )
                else:
                    per_action.append( conf[s] )
            final.append( np.array([max(per_action)]) )
        return final, None

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec , np.inf) ] )
        return W

    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        input_dim = self.d + (self.A-1) * self.d
        print('input dim', input_dim)

        self.net1 = Network_exploitation(input_dim, self.m).to(self.device)
        print(f'Net1 has {count_parameters(self.net1):,} trainable parameters.')

        self.net2 = Network_exploration(input_dim * 2, self.m).to(self.device)
        print(f'Net2 has {count_parameters(self.net2):,} trainable parameters.')

        self.contexts = {}
        for i in range(self.N):
            if self.eta[i]>0:
                self.contexts[i] =  {'V_it_inv': torch.eye(input_dim * 2).to(torch.float16).cpu() }

    def encode_context(self, X):
        X = torch.from_numpy(X).to(self.device)
        ci = torch.zeros(1, self.d).to(self.device)
        x_list = {}
        for a in range(self.A):
            inputs = []
            for l in range(a):
                inputs.append(ci)
            inputs.append(X)
            for l in range(a+1, self.A):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1).to(torch.float32)
            x_list[a] = inputs

        x_list_action = {}
        for k in range(self.N):
            x_list_action[k] = []
            for a in np.unique(self.game.FeedbackMatrix[k]) :
                x_list_action[k].append( x_list[a] )
       
        return x_list, x_list_action


    def get_action(self, t, X):

        self.X = X
        halfspace = []
        self.x_list, self.x_list_action = self.encode_context(X)

        self.f1_list, self.f2_list, self.dc_list, self.index = {}, {}, {}, {}
        for a in range(self.N):
            self.f1_list[a] = []
            self.f2_list[a] = [] 
            self.dc_list[a] = []

            for x in self.x_list_action[a]:
                f1_k, f2_k, dc_k = EE_forward(self.net1, self.net2, x )
                self.f1_list[a].append(f1_k)
                self.f2_list[a].append(f2_k)
                self.dc_list[a].append(dc_k)    
                # q[k] = f1_k 
                # w[k] = f2_k

            self.index[a] = np.argmax( self.f2_list[a] ) if len( np.unique(self.game.FeedbackMatrix[a]) ) > 1 else None
        
        q = [ np.array(value) for value in self.f1_list.values()]
        w = [ np.array( max(value) ) for value in self.f2_list.values()]
        # print( 'estimate', q )
        # print('conf   ', w )
        # print('index', self.index)

        #print('########################### eliminate actions')
        for pair in self.mathcal_N:
            tdelta, c = 0, 0
            for k in  self.V[ pair[0] ][ pair[1] ]:
                #print('k', k, 'pair ', pair, 'v ', self.v[ pair[0] ][ pair[1] ][k].T.shape , 'q[k] ', q[k].shape  )
                tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] #* np.sqrt( (self.d+1) * np.log(t) ) * self.d
            #print('pair', pair, 'tdelta', tdelta, 'confidence', c)
            # print('pair', pair,  'tdelta', tdelta, 'c', c, 'sign', np.sign(tdelta)  )
            # print('sign', np.sign(tdelta) )
            #tdelta = tdelta[0]
            #c = np.inf
            if( abs(tdelta) >= c):
                halfspace.append( ( pair, np.sign(tdelta) ) ) 
            
        
        # print('########################### compute halfspace')
        # print('halfspace', halfspace)
        #print('##### step1')
        code = self.halfspace_code(  sorted(halfspace) )
        #print('##### step2')
        P_t = self.pareto_halfspace_memory(code, halfspace)
        #print('##### step3')
        if len(P_t)>1:
            N_t = self.neighborhood_halfspace_memory(code, halfspace)
            # N_t = [ [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [6, 7], [6, 8], [6, 9], [6, 10], [7, 8], [7, 9], [7, 10], [8, 9], [8, 10], [9, 10] ]
            #self.neighborhood_halfspace_memory(code,halfspace)
            #print(N_t)
        else:
            N_t = []
        print('P_t', len(P_t), P_t, 'N_t', N_t)
        
        # print('########################### rarely sampled actions')
        Nplus_t = []
        for pair in N_t:
            Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
        Nplus_t = np.unique(Nplus_t)
        V_t = []
        for pair in N_t:
            V_t.extend( self.V[ pair[0] ][ pair[1] ] )
        V_t = np.unique(V_t)
        R_t = []

        for k, eta in zip(V_t, self.eta):
            if eta>0:
                indx = self.index[k]
                feature = self.dc_list[k][indx]
                V_it_inv = self.contexts[k]['V_it_inv']
                V_it_inv = V_it_inv.to(self.device)
                V_feature = torch.matmul(V_it_inv, feature.t() )
                feature_V = torch.matmul(feature, V_it_inv)
                val =  torch.matmul(feature_V, V_feature).item()
                t_prime = t+2
                rate = self.eta[k] * t_prime**(2/3)  * ( self.alpha * np.log(t_prime) )**(1/3)  
                if val > 1/rate : 
                    R_t.append(k)

                V_it_inv = V_it_inv.cpu()
                del V_it_inv
                torch.cuda.empty_cache()

        # print('########################### play action')
        union1= np.union1d(  P_t, Nplus_t )
        union1 = np.array(union1, dtype=int)
        
        explored = 1 if len(union1)>=2 else 0

        
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


    def update(self, action, feedback, outcome, t, X):
        

        # print('### Update the Gram matrix:')
        #print('convert to numpy')
        if len( np.unique(self.game.FeedbackMatrix[action]) ) > 1:
            indx = self.index[action]
            feature = self.dc_list[action][indx]
        else:
            feature =  self.dc_list[action][0]

        # Usage
        if self.eta[action]>0:
            V_it_inv = self.contexts[action]['V_it_inv']
            V_it_inv = V_it_inv.to(self.device)
            self.contexts[action]['V_it_inv'] = sherman_morrison_update(V_it_inv, feature)
            V_it_inv = V_it_inv.cpu()
            del V_it_inv
            torch.cuda.empty_cache()

        # print('implement the formula')
        # V_it_inv = self.contexts[action]['V_it_inv']
        # self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ feature.T @ feature @ V_it_inv ) / ( 1 + feature @ V_it_inv @ feature.T ) 
        
        # print('### Update the parameters of the model:')
        sigma_k = len(np.unique(self.game.FeedbackMatrix[action]))
        #print('sigma_k', sigma_k)
        if sigma_k > 1:
            feedbacks = np.zeros( sigma_k )
            feedbacks[feedback] = 1
        else:
            feedbacks = np.ones(1)
        
        x_features = torch.cat( self.x_list_action[action] )
        self.X1_train.append( x_features.cpu() )
        self.y1.append( torch.Tensor(feedbacks).cpu()  )

        grad_features = torch.cat( self.dc_list[action] )
        self.X2_train.append( grad_features.cpu() )
        self.y2.append( torch.Tensor(feedbacks - self.f1_list[action] ).cpu() )
            
        global_loss = []
        global_losses = []

        if (t>self.N):

            if (t % 50 == 0 and t<1000) or (t % 500 == 0 and t>=1000):

                self.train_NN_batch(self.net1, self.X1_train, self.y1)
                self.train_NN_batch(self.net2, self.X2_train, self.y2)

        return global_loss, global_losses
                

    def train_NN_batch(self, model, X, y, num_epochs=10, lr=0.001, batch_size=64):
        model.train()
        X = torch.cat(X).float()
        y = torch.cat(y).float()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = X.size(0)

        for i in range(num_epochs):
            # print('epoch {}'.format(i))
            batch_loss = 0.0
            
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).view(-1)

                loss = torch.mean((pred - y) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num




    def halfspace_code(self, halfspace):
        string = ''
        for element in halfspace:
            pair, sign = element
            string += '{}{}{}'.format(pair[0],pair[1], sign)
        return string 


    def pareto_halfspace_memory(self, code, halfspace):

        result = self.memory_pareto.get(code)
        
        if result is None:

            result = geometry_pulp.getParetoOptimalActions(
                self.game.LossMatrix, 
                self.N, 
                self.M, 
                halfspace, 
                self.num_workers  ) 
            self.memory_pareto[code] = result

        return result

    def neighborhood_halfspace_memory(self, code, halfspace):

        result = self.memory_neighbors.get(code)
        

        if result is None:
            print('step 3 b')
            result = geometry_pulp.getNeighborhoodActions(
                self.game.LossMatrix, 
                self.N, 
                self.M, 
                halfspace, 
                self.mathcal_N, 
                self.num_workers
            )
            self.memory_neighbors[code] = result

        return result
