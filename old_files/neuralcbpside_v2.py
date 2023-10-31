import numpy as np
import geometry_gurobi

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim

import itertools
from functorch import vmap, vjp
from functools import partial

def get_combinations(A):
    identity_matrix = torch.eye(A)
    combinations = list(itertools.combinations(identity_matrix, A))[0]
    return torch.stack(combinations).cuda()

class Network(nn.Module):
    def __init__(self, output_dim, dim, hidden_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def convert_list(A):
    B = []
    B.append(np.array([A[0]]).reshape(1, 1))
    sub_array = np.array(A[1:]).reshape(2, 1)
    B.append(sub_array)
    return B

class NeuralCBPside():

    def __init__(self, game, d, alpha, lbd, hidden):

        self.name = 'NeuralCBPsidev3'

        self.game = game
        self.d = d
        self.N = game.n_actions
        self.M = game.n_outcomes

        self.SignalMatrices = game.SignalMatrices
        self.pareto_actions = geometry_gurobi.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        self.N_plus =  game.N_plus
        self.V = game.V
        self.v = game.v 
        self.W = self.getConfidenceWidth( )
        self.eta =  self.W **2/3 
        self.A = geometry_gurobi.alphabet_size(game.FeedbackMatrix_PMDMED, game.N, game.M)

        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.hidden = hidden
        self.m = hidden

        self.lbd = lbd
        self.alpha = alpha

        self.g_list = []
        self.functionnal = []
        for i in range(self.N):
            output_dim = len( set(self.game.FeedbackMatrix[i]) )
            func = Network( output_dim, self.d, hidden_size=self.hidden).cuda()
            total_params = sum(p.numel() for p in func.parameters() if p.requires_grad)
            self.functionnal.append( {'features':[], 'labels':[], 
                                      'V_it_inv': np.identity(self.d),
                                      'weights': func, 
                                      'p': total_params,
                                      'Z_it_inv':self.lbd * torch.eye(total_params).cuda() } )
    def set_nlabels(self, nlabels):
        self.d = nlabels

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec ) ] )
        return W

    def reset(self,):
        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)]  #[ np.zeros(    len( np.unique(self.game.FeedbackMatrix[i] ) )  ) for i in range(self.N)] 
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.g_list = []
        self.functionnal = []
        for i in range(self.N):
            output_dim = len( set(self.game.FeedbackMatrix[i]) )
            func = Network( output_dim, self.d, hidden_size=self.hidden).cuda()
            total_params = sum(p.numel() for p in func.parameters() if p.requires_grad)
            self.functionnal.append( {'features':[], 'labels':[], 
                                      'V_it_inv': np.identity(self.d),
                                      'weights': func, 
                                      'p': total_params,
                                      'Z_it_inv':self.lbd * torch.eye(total_params).cuda() } )
    def get_action(self, t, X):

        if t < self.N: # jouer chaque action une fois au debut du jeu
            action = t

        else: 
            
            self.g_list = []
            halfspace = []
            q = []
            w = []
                        
            for i in range(self.N):
                
                self.functionnal[i]['weights'].zero_grad()
                pred =  self.functionnal[i]['weights']( torch.from_numpy(X.T).float().cuda() ) 
                # print('pred initial', pred)
                
                if pred.shape[1] == 1:
                    
                    pred.backward()
                    g = torch.cat([p.grad.flatten().detach() for p in self.functionnal[i]['weights'].parameters() ])
                    self.g_list.append(g)

                else: # gradient with respect to each element in the predicted vector

                    output_dim = len( set(self.game.FeedbackMatrix[i]) )
                    sum = torch.sum(pred)
                    self.functionnal[i]['weights'].zero_grad()
                    sum.backward( retain_graph = True )
                    g = torch.cat( [ p.grad.flatten().detach() for p in self.functionnal[i]['weights'].parameters() ] )
                    self.g_list.append(g)

                width2 = (g.T @ self.functionnal[i]['Z_it_inv'] @ g) / self.m
                width = torch.sqrt( width2 )
                
                sigma_i = len(self.SignalMatrices[i])
                factor = sigma_i * (  np.sqrt(  self.functionnal[i]['p'] * np.log(t) + 2 * np.log(1/t**2)   ) + np.sqrt(self.lbd) * sigma_i )
                formule = factor * width
                print('factor', factor, 'width', width)
                w.append( formule.cpu().detach().numpy() )
                q.append( pred.cpu().detach().numpy().T )

            print('confidence', w)
            print('estimates', q)
                
            for pair in self.mathcal_N:
                tdelta = np.zeros( (1,) )
                c = 0

                for k in  self.V[ pair[0] ][ pair[1] ]:
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k] ) * w[k] 

                tdelta = tdelta[0]
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

        return action

    def update(self, action, feedback, outcome, t, X):

        V_it_inv = self.functionnal[action]['V_it_inv']
        self.functionnal[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 

        if t >= self.N:
            z = torch.unsqueeze( self.g_list[action] , 1) 
            # print(z.shape, z)
            Z_it_inv = self.functionnal[action]['Z_it_inv']
            self.functionnal[action]['Z_it_inv'] = Z_it_inv - ( Z_it_inv @ z @ z.T @ Z_it_inv ) / ( 1 + z.T @ Z_it_inv @ z ) 


        e_y = np.zeros( (self.M, 1) )
        e_y[outcome] = 1
        Y_t =  self.game.SignalMatrices[action] @ e_y 

        self.functionnal[action]['labels'].append( torch.from_numpy(Y_t.T).float() )
        self.functionnal[action]['features'].append( torch.from_numpy(X.T).float() )

        optimizer = optim.SGD(self.functionnal[action]['weights'].parameters(), lr=1e-2, weight_decay=self.lbd)
        length = len(self.functionnal[action]['labels'])
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0

        while True:
            batch_loss = 0
            for idx in index:
                c = self.functionnal[action]['features'][idx]
                f = self.functionnal[action]['labels'][idx].cuda()
                pred = self.functionnal[action]['weights']( c.cuda() )
                # print(c, f, pred)
                optimizer.zero_grad()
                # print(c.shape)
                loss = nn.MSELoss()(pred, f)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 100:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length



        

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

