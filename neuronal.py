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

    def __init__(self, model, budget, num_cls, margin, m, device):
        self.name = 'neuronal'
        
        self.device = device
        self.model = model
        self.num_cls = num_cls
        self.m = m
        self.budget = budget
        self.query_num = 0
        self.margin = margin #according to their parameter search
        self.N = num_cls+1

    def reset(self, d):

        self.d = d

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        if self.model == 'MLP':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_MLP(input_dim, output_dim,  self.m).to(self.device)
            

            exp_dim = 1660 if self.num_cls==10 else 1644 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)


        elif self.model == 'LeNet':
            input_dim = self.d
            output_dim = self.num_cls
            self.net1 = EENets.Network_exploitation_LeNet(output_dim, ).to(self.device)

            exp_dim = 1330 if self.num_cls==10 else 1317 
            output_dim = self.num_cls
            self.net2 = EENets.Network_exploration(exp_dim, output_dim, self.m).to(self.device)


    def get_action(self, t, X):

        print('X shape', X.shape)
        
        self.X = X.to(self.device)
        self.f1, self.f2, self.dc = EENets.EE_forward(self.net1, self.net2, self.X)
        u = self.f1[0] + 1 / (self.query_num+1) * self.f2
        print('u', u)
        u_sort, u_ind = torch.sort(u)
        i_hat = u_sort[-1]
        i_deg = u_sort[-2]

        explored = 0
        if abs(i_hat - i_deg) < self.margin * 0.1:
            explored = 1 
        else:
            explored = 0

        self.pred = int(u_ind[-1].item()) +1
        print('pred',self.pred)
        action = self.pred if explored == 0 else 0

        if t<self.N:
            action = t

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
    
    def update(self, action, feedback, outcome, t, X):

        lbl = outcome #+1 

        if (action == 0) and (self.query_num < self.budget): 
            self.query_num += 1

            self.X1_train.append(self.X)
            self.X2_train.append( self.dc )
            r_1 = torch.zeros(self.num_cls).to(self.device)
            r_1[lbl] = 1
            self.y1.append(r_1) 
            self.y2.append((r_1 - self.f1)[0])
            
        if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): #
            # print('X1_train',self.X1_train)
            self.train_NN_batch(self.net1, self.X1_train, self.y1)
            self.train_NN_batch(self.net2, self.X2_train, self.y2)

        return None, None
        

    def train_NN_batch(self, model, hist_X, hist_Y, num_epochs=10, lr=0.001, batch_size=64):
        model.train()

    
        hist_X = torch.cat(hist_X).float()
        hist_Y = torch.stack(hist_Y).float().detach()
        # print(X.shape, Y.shape)

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
                print('loss', loss)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num
        