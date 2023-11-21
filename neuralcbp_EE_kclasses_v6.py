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
from skimage.measure import block_reduce


class Network_exploitation(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        latent = self.fc1(x)
        x = self.activate(latent)
        x = self.fc2(x)
        return x, latent
    
class Network_exploration(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=100):
        super(Network_exploration, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        latent = self.fc1(x)
        x = self.activate(latent)
        x = self.fc2(x)
        return x, None

def EE_forward(game, net1, net2, x):
    x.requires_grad = True
    f1, latent = net1(x)
    net1.zero_grad()

    f1.sum().backward(retain_graph=True)

    grad = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    #dc = dc / torch.linalg.norm(dc)
    grad = block_reduce(grad.cpu(), block_size=51, func=torch.mean)
    # grad = grad.to(x.device)
    # print('grad', grad.shape)
    # print('latent', latent[0].shape)

    dc = torch.cat([grad, latent[0] ]  )
    dc = dc / torch.linalg.norm(dc)
    # print('dc', dc.shape)
    f2, _ = net2(dc)

    return f1, f2, dc.unsqueeze(0)



def sherman_morrison_update(V_it_inv, feature):
    
    V_feature = torch.matmul(V_it_inv, feature.t())
    feature_V = torch.matmul(feature, V_it_inv)
    denom = 1 + torch.matmul(feature_V, feature.t())
    V_it_inv -= torch.matmul(V_feature, feature_V) / denom

    return V_it_inv
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CBPside():

    def __init__(self, game, alpha, m, num_cls, device):

        self.name = 'neuralCBPside'
        self.device = device
        
        self.num_workers = 1 #int ( os.environ.get('SLURM_CPUS_PER_TASK', default=1) )
        print('num workers', self.num_workers  )

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = sum( [ len( np.unique( game.FeedbackMatrix[i] ) ) for i in range(self.N)] )

        self.SignalMatrices = game.SignalMatrices

        self.pareto_actions = geometry_pulp.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [], self.num_workers)
        self.mathcal_N = game.mathcal_N

        self.N_plus =  game.N_plus

        self.V = game.V

        self.v = game.v 

        self.W = self.getConfidenceWidth( )

        self.alpha = alpha
            

        self.eta =  self.W**(2/3) 
        self.m = m

        self.num_cls = num_cls
        self.err_counter = 0

    def convert_pred_format(self,pred):
        print('pred', pred)
        final = []
        for k in range(self.game.N):
            print('k', k)
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                print('s', s)
                if s in self.game.informative_symbols:
                    per_action.append( pred[0][s].detach().cpu() )
                else:
                    per_action.append( 1 )
            final.append( np.array(per_action) )
        return final

    def convert_conf_format(self, conf, ):
        final = []
        for k in range(self.game.N):
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                if s in self.game.informative_symbols:
                    per_action.append( conf[s].detach().cpu()  )
                else:
                    per_action.append( 0 )
            final.append( np.array([max(per_action)]) )
        return final

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            # print(pair)
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec , np.inf) ] )
        return W

    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        input_dim = self.m + self.m * self.A
        print('input dim', input_dim)

        self.net1 = Network_exploitation(self.game, self.d, self.m).to(self.device)
        print(f'Net1 has {count_parameters(self.net1):,} trainable parameters.')

        self.net2 = Network_exploration(self.game, input_dim, self.m).to(self.device)
        print(f'Net2 has {count_parameters(self.net2):,} trainable parameters.')

        self.contexts = {}
        for i in range(self.N):
            self.contexts[i] =  {'V_it_inv': torch.eye(input_dim),
                                 'X1_train': [],
                                 'y1_train': [],
                                 'X2_train': [],
                                 'y2_train': []  }


    def get_action(self, t, X):

        self.X = torch.from_numpy(X).to(self.device)
        halfspace = []

        self.f1, self.f2, self.dc = EE_forward(self.game, self.net1, self.net2, self.X )

        
        q = self.convert_pred_format(self.f1)
        w = self.convert_conf_format(self.f2)
        print('pred', q)
        print('conf', w )
        # print('dc', self.dc)

        #print('########################### eliminate actions')
        for pair in self.mathcal_N:
            tdelta, c = 0, 0
            for k in  self.V[ pair[0] ][ pair[1] ]:
                tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k], np.inf ) * w[k]

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
        elif len(P_t) == 0:
            P_t = [i for i in range(self.N)]
            N_t = self.mathcal_N
            self.err_counter += 1
            print('there was an estimation error')
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

        # print('V_t', V_t)
        R_t = []
        for k, eta in zip(V_t, self.eta):
            V_it_inv = self.contexts[k]['V_it_inv'].to(self.device)
            # print('shapeeee',V_it_inv.shape, self.dc.t().shape, self.dc.shape)
            V_feature = torch.matmul(V_it_inv, self.dc.t() )
            feature_V = torch.matmul(self.dc, V_it_inv)
            val =  torch.matmul(feature_V, V_feature).item()
            t_prime = t+2
            rate = self.eta[k] * t_prime**(2/3)  * ( self.alpha * np.log(t_prime) )**(1/3)  
            if val > 1/rate : 
                R_t.append(k)

        # print('########################### play action')
        union1= np.union1d(  P_t, Nplus_t )
        union1 = np.array(union1, dtype=int)
        
        
        print('union1', union1, 'R', R_t)
        S =  np.union1d(  union1  , R_t )
        S = np.array( S, dtype = int)
        # print('S', S)
        S = np.unique(S)
        # print()
        values = { i:self.W[i]*w[i] for i in S}
        print('value', values)
        action = max(values, key=values.get)

        if t<self.N:
            action = t

        history = {'monitor_action':action,  }
            
        return action, history


    def update(self, action, feedback, outcome, t, X):


        ### UPDATE PSEUDO-COUNTS:
        V_it_inv = self.contexts[action]['V_it_inv']
        V_it_inv = V_it_inv.to(self.device)
        self.contexts[action]['V_it_inv'] = sherman_morrison_update(V_it_inv, self.dc)


        ### UPDATE HISTORY OF OBSERVATIONS

        sigma_i = len(np.unique(self.game.FeedbackMatrix[action]))
        if sigma_i>1:
            feedbacks = np.zeros( sigma_i )
            feedbacks[feedback] = 1
        else:
            feedbacks = np.ones( 1 )

        self.contexts[action]['X1_train'].append( self.X )
        self.contexts[action]['y1_train'].append( torch.Tensor(feedbacks).unsqueeze(0) )

        self.contexts[action]['X2_train'].append( self.dc )
        self.contexts[action]['y2_train'].append(  torch.Tensor(feedbacks - self.f1[action] ).unsqueeze(0) )


        self.X2_train.append( self.dc )
        self.y2.append( torch.Tensor(feedbacks - self.f1[action] ).cpu() )
            
        global_loss = []
        global_losses = []

        if (t>self.N):

            if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): 

                self.train_NN_batch(self.net1, 'X1_train', 'y1_train' )
                self.train_NN_batch(self.net2, 'X2_train', 'y2_train' )

        return global_loss, global_losses

    def train_NN_batch(self, model, target_a,target_b, num_epochs=10, lr=0.001, batch_size=64):

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        

        for i in range(num_epochs):

            batch_loss = 0.0
            
            for k in range(self.N):
                if self.eta[k]>0:
                    X = torch.cat(self.contexts[k][target_a]).float().to(self.device)
                    y = torch.cat(self.contexts[k][target_b]).float().to(self.device)
                    # print('size of tensors', X.shape, y.shape)
                    dataset = TensorDataset(X, y)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    # num = X.size(0)

                    for x, y in dataloader:
                        x, y = x.to(self.device), y.to(self.device)
                        pred, _ = model(x) #.view(-1)
                        loss = torch.mean((pred[k] - y) ** 2)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        batch_loss += loss.item()
                    
                    # if batch_loss / num <= 1e-3:
                    #     return batch_loss / num

        return batch_loss #/ num




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
