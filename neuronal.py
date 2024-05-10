import os
import time
import math
import random
import numpy as np

import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import EENets


class NeuronAL():


    def __init__(self, model, context_type, budget, num_cls, margin, unit_loss, m, device):
        self.name = 'neuronal'
        
        self.device = device
        self.model = model
        self.num_cls = num_cls
        self.m = m
        self.unit_loss = unit_loss
        self.budget = budget
        self.query_num = 0
        self.margin = margin #according to their parameter search
        self.N = num_cls+1
        self.context_type = context_type
        # self.batch == 0

    def predictor(self,X,y):
        if self.context_type == 'CIFAR10' and self.model == 'LeNet':
            # X = torch.unsqueeze(X, 1)
            print(X.shape)
            y_pred, _ = self.net1(X)
        elif self.context_type in ['MNIST', 'FASHION'] and self.model == 'LeNet':
            X = torch.squeeze(X, 0)
            X = torch.unsqueeze(X, 1)
            y_pred, _ = self.net1(X)
        else:
            y_pred, _ = self.net1(X)
        return y_pred

    def reset(self, d):

        self.d = d

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []
        # self.batch == 0

        
        if self.context_type =='MNISTbinary' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 1644 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type in ['MNIST', 'FASHION'] and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 1660 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type == 'LETTERS' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 1691 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type == 'adult' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 312 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type == 'MagicTelescope' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 126 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type == 'covertype' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 308 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

        elif self.context_type == 'shuttle' and self.model == 'MLP':
            self.size = 51
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            exp_dim = 134 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)

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

        # print('X shape', X.shape)
        
        self.X = X.to(self.device)

        self.f1, self.f2, self.dc = EENets.EE_forward(self.net1, self.net2, self.X, self.size)
        # the division by t , or query number was not found to change the performance 
        # and does not appear in the official pseudo code
        u = self.f1[0] + self.f2 #1 / (self.query_num+1) *
        # print('u', u)
        u_sort, u_ind = torch.sort(u)
        i_hat = u_sort[-1]
        i_deg = u_sort[-2]

        explored = 0
        if abs(i_hat - i_deg) < self.margin * 0.1:
            explored = 1 
        else:
            explored = 0

        self.pred = int(u_ind[-1].item()) +1
        # print('pred',self.pred)
        action = self.pred if explored == 0 else 0

        if t<self.N:
            action = t

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
    
    def update(self, action, feedback, outcome, t, X, lossmatrix):

        lbl = outcome 

        if (action == 0) and (self.query_num < self.budget): 
            self.query_num += 1

            self.X1_train.append(self.X)
            self.X2_train.append( self.dc )
            r_1 = torch.zeros(self.num_cls).to(self.device)
            r_1[lbl] = 1 if self.unit_loss==True else lossmatrix[ self.pred ][ outcome ]
            self.y1.append(r_1) 
            self.y2.append((r_1 - self.f1)[0])
            

        if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): #
        # print('X1_train',self.X1_train)
        # if action == 0 and (t>self.N):
            self.train_NN_batch(self.net1, self.X1_train, self.y1)
            self.train_NN_batch(self.net2, self.X2_train, self.y2)

        # if action == 0:
        #     self.batch = self.batch + 1

        # if action == 0 and (t>self.N) and self.batch == 10:
        #     self.train_NN_batch(self.net1, self.X1_train, self.y1 )
        #     self.train_NN_batch(self.net2, self.X2_train, self.y2 )
        #     self.batch == 0

        return None, None
        

    def train_NN_batch(self, model, hist_X, hist_Y, num_epochs=40, lr=0.001, batch_size=64):
        model.train()

        hist_X = torch.cat(hist_X).float()
        hist_Y = torch.stack(hist_Y).float().detach()
        # print(hist_X.shape, hist_Y.shape)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(hist_X, hist_Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # num = X.size(1)
        num = hist_X.size(1)

        for i in range(num_epochs):
            batch_loss = 0.0
            
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                # y = torch.reshape(y, (1,-1))
                pred, _ = model(x)
                # pred = pred.view(-1)
                # print(pred.shape, y.shape)

                optimizer.zero_grad()
                loss = torch.mean((pred - y) ** 2)
                # print('loss', loss)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num
        