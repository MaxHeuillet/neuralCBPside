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


import EENets

############################################################ ############################################################ 
############################################################  Agent CBP:
############################################################ ############################################################ 


def sherman_morrison_update(V_it_inv, feature):
    
    V_feature = torch.matmul(V_it_inv, feature.t())
    feature_V = torch.matmul(feature, V_it_inv)
    denom = 1 + torch.matmul(feature_V, feature.t())
    V_it_inv -= torch.matmul(V_feature, feature_V) / denom

    return V_it_inv
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CBPside():

    def __init__(self, game, context_type, model, alpha, m, num_cls, device):

        self.name = 'neuralCBPside'
        self.device = device
        self.model = model
        self.context_type = context_type
        
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
        # print('pred', pred)
        final = []
        for k in range(self.game.N):
            # print('k', k)
            per_action = []
            for s in np.unique(self.game.FeedbackMatrix[k]):
                # print('s', s)
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
                    per_action.append( 1 )
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

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []
        self.memory_pareto = {}
        self.memory_neighbors = {}
        
        if self.context_type == 'MNISTbinary' and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 1644 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        if self.context_type in ['MNIST', 'FASHION'] and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 1660 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type == 'adult' and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 312 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type == 'MagicTelescope' and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 126 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type == 'covertype' and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 308 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type == 'shuttle' and self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 134 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type == 'CIFAR10' and self.model == 'LeNet':
            output_dim = self.num_cls
            channels = 3
            print(channels)
            latent_dim = 1200
            self.net1 = EENets.Network_exploitation_LeNet(latent_dim,channels, output_dim,  ).to(self.device)
            self.size = 153
            exp_dim = 3948 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }

        elif self.context_type in ['MNIST', 'FASHION'] and self.model == 'LeNet':
            input_dim = self.d
            output_dim = self.num_cls
            self.size = 51
            channels = 1
            latent_dim = 256
            self.net1 = EENets.Network_exploitation_LeNet(latent_dim, channels, output_dim, ).to(self.device)
            exp_dim = 992 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

            self.contexts = {}
            for i in range(self.N):
                self.contexts[i] =  {'V_it_inv': torch.eye(exp_dim)  }




    def get_action(self, t, X):

        self.X = X.to(self.device)
        halfspace = []

        self.f1, self.f2, self.dc = EENets.EE_forward(self.net1, self.net2, self.X, self.size )
        q = self.convert_pred_format(self.f1)
        w = self.convert_conf_format(self.f2)

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
        print('feedback', feedback, self.game.informative_symbols)
        if feedback in self.game.informative_symbols:

            feedbacks = torch.zeros( len(self.game.informative_symbols) ).to(self.device)
            feedbacks[feedback] = 1
            self.X1_train.append( self.X )
            self.y1.append( torch.Tensor(feedbacks).unsqueeze(0) )

            self.X2_train.append( self.dc )
            self.y2.append(  torch.Tensor(feedbacks - self.f1.detach() ) )
            
        global_loss = []
        global_losses = []

        if (t>self.N):

            if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000):
                print('train 1')
                self.train_NN_batch(self.net1, self.X1_train, self.y1 )
                print('train 2')
                self.train_NN_batch(self.net2, self.X2_train, self.y2 )

        return global_loss, global_losses

    def train_NN_batch(self, model, hist_X, hist_Y, num_epochs=40, lr=0.001, batch_size=64):

        model.train()
        # print(len(hist_X), len(hist_Y) )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        hist_X = torch.cat(hist_X).float().to(self.device)
        hist_Y = torch.cat(hist_Y).float().to(self.device)
        # print(hist_X.shape, hist_Y.shape )

        dataset = TensorDataset(hist_X, hist_Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = hist_X.size(1)


        for i in range(num_epochs):

            batch_loss = 0.0

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = model(x) #.view(-1)
                # print('pred', pred)
                # print('y', y)
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
