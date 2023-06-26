import numpy as np
import geometry_v3

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize

# import itertools

class Network(nn.Module):
    def __init__(self, output_dim, dim, hidden_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        x = self.fc2(self.activate(self.fc1(x)))
        return x
    
class Network2(nn.Module):
    def __init__(self, output_dim, dim, hidden_size=10):
        super(Network2, self).__init__()
        self.linear = nn.Linear(dim, output_dim)
    def forward(self, x):
        x = self.linear(x)
        return x

# class Network3(nn.Module):
#     def __init__(self, output_dim, dim, hidden_size=10):
#         super(Network3, self).__init__()
#         self.fc1 = nn.Linear(dim, hidden_size)
#         self.dropout = nn.Dropout(p=0.2)
#         self.activate = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_dim)
    
#     def forward(self, x):
#         x = self.dropout(self.activate(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# class Network(nn.Module):
#     def __init__(self, output_dim, dim, hidden_size=10):
#         super(Network, self).__init__()
#         self.fc1 = nn.Linear(dim, hidden_size)
#         self.activate = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_dim)
#         self.sigmoid = nn.Sigmoid( )
#     def forward(self, x):
#         x = self.fc2(self.activate(self.fc1(x)))
#         x = self.sigmoid(x)
#         return x

class NeuralCBPside():

    def __init__(self, game, horizon, factor_choice, alpha, lbd, hidden, device):

        self.name = 'neuralcbpside'
        self.device = device
        self.horizon = horizon

        self.game = game
        
        self.N = game.n_actions
        self.M = game.n_outcomes

        self.SignalMatrices = game.SignalMatrices
        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        self.N_plus =  game.N_plus
        self.V = game.V
        self.v = game.v 
        self.W = self.getConfidenceWidth( )
        self.eta =  self.W **2/3 
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix_PMDMED, game.N, game.M)

        self.m = hidden
        self.lbd = lbd
        self.alpha = alpha
        self.factor_choice = factor_choice

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec, np.inf ) ] )
        return W

    def reset(self, d):
        self.d = d
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.counter = 0

        self.g_list = []
        self.features = None
        self.labels = None
        self.functionnal = []
        self.func = Network( 1, self.d * self.A, hidden_size=self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.p = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        self.d_init = np.random.normal(0, 0.01, self.p).reshape(-1, 1)
        self.detZt = self.lbd**self.p
        self.Z_t = self.lbd * np.identity(self.p)
        self.Z_t_inv = self.lbd * np.identity(self.p)

    
    def gamma_t(self, i, t, ):
        
        delta = 0.05 
        gamma_t = 2 * np.log( 2 * self.horizon * self.A / delta )

        return gamma_t

    def get_action(self, t, X):
        # print('self func0', self.func0)

        if t < self.N: # jouer chaque action une fois au debut du jeu
            action = t
            tdelta = 0
            factor = 0
            history = [t, np.nan, np.nan]

        else: 
            
            self.g_list = []
            g_buffer = {i: [] for i in range(self.N)}
            pred_buffer = {i: [] for i in range(self.N)}
            halfspace = []
            q = []
            w = []

            sigmas = [ len(self.SignalMatrices[i]) for i in range(self.N) ]

            # Get the unique elements of the matrix
            unique_elements = [0, 2, 1] #np.unique(self.game.FeedbackMatrix)
            for feedback in unique_elements:
                act_to_idx = np.where(self.game.FeedbackMatrix == feedback)[0][0]

                context = torch.from_numpy( X[feedback] ).float().to(self.device) 
                pred =  self.func( context)
                pred0 =  self.func0( context )

                self.func0.zero_grad()

                pred0.backward() 

                g = torch.cat([p.grad.flatten().detach() for p in self.func0.parameters()])
                g = torch.unsqueeze(g , 1)

                g_buffer[act_to_idx].append(g.cpu().detach().numpy() )
                pred_buffer[act_to_idx].append(pred.cpu().detach().numpy())
            
            self.g_list = [ np.mean(g_buffer[idx],0) if s>1 else g_buffer[idx][0] for idx, s in enumerate(sigmas) ] 
            
            # print('pred_buffer', pred_buffer)
            for i in range(self.N):
                width2 = (self.g_list[i].T @ self.Z_t_inv @ self.g_list[i]) / self.m
                width = np.sqrt( width2 )

                sigma_i = len(self.SignalMatrices[i])

                factor = self.gamma_t(i, t)

                formule = factor * width

                print('factor',factor,  'width', width,  )
                w.append( formule )
                q.append( np.array(pred_buffer[i]).reshape(sigma_i,1) )

            # print('confidence', w)
            print('estimates', q)
                
            for pair in self.mathcal_N:
                tdelta = np.zeros( (1,) )
                c = 0

                for k in  self.V[ pair[0] ][ pair[1] ]:
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k] 

                tdelta = tdelta[0]
                c = c[0][0]
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta) ) ) 

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
            #   val =  X.T @ self.functionnal[k]['V_it_inv'] @ X
            #   t_prime = t
            #   with np.errstate(divide='ignore'): 
            #     rate = np.sqrt( self.eta[k] * self.N**2 * 4 *  self.d**2  *(t_prime**(2/3) ) * ( self.alpha * np.log(t_prime) )**(1/3) ) 
            #     if val[0][0] > 1/rate : 
            #         R_t.append(k)
    
            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)
            print('union1', union1)

            S =  np.union1d(  union1  , R_t )
            S = np.array( S, dtype = int)
            S = np.unique(S)

            values = { i:self.W[i]*w[i] for i in S}
            action = max(values, key=values.get)

            history = [ action, factor, tdelta ]

        return action, history 

    def update(self, action, feedback, outcome, t, X):

        if t >= self.N:

            z = self.g_list[action]  
            
            Z_t = self.Z_t
            Z_t += z @ z.T / self.m
            self.Z_t = Z_t

            Z_t_inv = self.Z_t_inv
            self.Z_t_inv = Z_t_inv - ( Z_t_inv @ z @ z.T @ Z_t_inv ) / ( 1 + z.T @ Z_t_inv @ z ) 


        e_y = np.zeros( (self.M,1) )
        e_y[outcome] = 1
        Y_t = self.game.SignalMatricesAdim[action] @ e_y 

        self.features = X if self.features is None else np.concatenate((self.features, X), axis=0)
        self.labels = Y_t if self.labels is None else np.concatenate((self.labels, Y_t), axis=0)

        # print(self.features)
        # print(self.labels)

        optimizer = optim.SGD(self.func.parameters(), lr=0.01, weight_decay=self.lbd) 
        length = self.labels.shape[0]
        X_tensor = torch.tensor(self.features)
        y_tensor = torch.tensor(self.labels)
        dataset = TensorDataset(X_tensor, y_tensor)

        if length < 1000:
            dataloader = DataLoader(dataset, batch_size=length, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

        train_loss = self.epoch(dataloader, self.func, optimizer)
        # print(train_loss)

    def epoch(self,loader, model, opt=None):
        #""Standard training/evaluation epoch over the dataset"""
        expected_dtype = next(model.parameters()).dtype
        for X,y in loader:
            X,y = X.to(self.device), y.to(self.device)
            y = y.to(dtype=expected_dtype)
            X = X.to(dtype=expected_dtype)
            yp = model(X)
            loss = nn.MSELoss()(yp,y)
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

# while True:
#     batch_loss = 0
#     for idx in index:
#         c = torch.tensor(self.features[idx]).to(self.device)
#         c = c.to(dtype=expected_dtype)
#         f = torch.tensor(self.labels[idx]).to(self.device)
#         f = f.float()
#         pred = self.func( c )
#         # print('pred', pred.shape, c.shape, f.shape)
#         optimizer.zero_grad()
#         loss = nn.MSELoss()(pred, f)
#         # loss = nn.BCELoss()(pred, f)
#         loss.backward()
#         optimizer.step()
#         batch_loss += loss.item()
#         tot_loss += loss.item()
#         cnt += 1
#         if cnt >= 100:
#             return tot_loss / 100
#     if batch_loss / length <= 1e-3:
#         return batch_loss / length




     
