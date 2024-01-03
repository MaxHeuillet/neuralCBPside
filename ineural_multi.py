import os
import time
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
        
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
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
    # print('dc shape', dc.shape)
    f2 = net2(dc)
    return f1.item(), f2.item(), dc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class INeurALmulti():

    def __init__(self, budget, num_cls, margin, m, device):
        self.name = 'ineuralmulti'
        
        self.device = device
        self.m = m
        self.num_cls = num_cls

        self.budget = budget
        self.query_num = 0
        self.margin = margin #according to their parameter search
        self.N = num_cls+1
        self.ber = 1.1

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

    def predictor(self,X,y):
        y_preds = []
        for x in X:
            x = x.unsqueeze(0)
            # print('x shape', x.shape)
            x_list = self.encode_context(x)
            f1_list = []
            for k in range(self.num_cls):
                y_pred = self.net1(x_list[k])
                f1_list.append( y_pred.item() )
            y_preds.append(f1_list)
        return torch.Tensor(y_preds)

    def reset(self, d):

        self.d = d

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        input_dim = self.d + (self.num_cls-1) * self.d
        print('input dim', input_dim)

        self.net1 = Network_exploitation(input_dim, self.m).to(self.device)
        self.net2 = Network_exploration(input_dim * 2, self.m).to(self.device)

        print(f'Net1 has {count_parameters(self.net1):,} trainable parameters.')
        print(f'Net2 has {count_parameters(self.net2):,} trainable parameters.')

    def encode_context(self, X):
        X = X.to(self.device)
        ci = torch.zeros(1, self.d).to(self.device)
        x_list = []
        for k in range(self.num_cls):
            inputs = []
            for l in range(k):
                inputs.append(ci)
            inputs.append(X)
            for l in range(k+1, self.num_cls):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1).to(torch.float32)
            x_list.append(inputs)
        return x_list

    def get_action(self, t, X):

        print('X shape', X.shape)

        self.x_list = self.encode_context(X)
        # print('xlist[0] shape', x_list[0].shape)
        self.f1_list, self.f2_list, self.dc_list, self.u_list = [], [], [], []
        prob = -1
        
        for k in range(self.num_cls):
            f1_k, f2_k, dc_k = EE_forward(self.net1, self.net2, self.x_list[k])
            u_k = f1_k + 1 / (t+1) * f2_k
            self.f1_list.append(f1_k)
            self.f2_list.append(f2_k)
            self.dc_list.append(dc_k)
            self.u_list.append((k, u_k))
            if u_k > prob:
                prob = u_k
                self.pred = k

        self.u_list = sorted(self.u_list, key=lambda x: x[1], reverse=True)
        print('ulist', self.u_list)
        self.pred = self.u_list[0][0] + 1
        print('pred',self.pred)
        
        diff = self.u_list[0][1] - self.u_list[1][1]
        print('diff', diff)
        if diff < self.margin * 0.1:
            explored = 1
        else:
            explored = 0

        action = self.pred if explored == 0 else 0
        print('action', action)
        if t<self.N:
            action = t

        print('action2', action)

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
        

    def update(self, action, feedback, outcome, t, X, loss):

        lbl = outcome +1 
        if self.pred != lbl:
            reward = 0
        else:
            reward = 1

        if (action == 0) and (self.query_num < self.budget): 
            self.query_num += 1

            for k in range(self.num_cls):
                k_prime = k+1
                if k_prime != self.pred and k_prime != lbl:
                    continue
                self.X1_train.append( self.x_list[k].detach().cpu() )
                self.X2_train.append( self.dc_list[k].detach().cpu() )
                if k_prime == self.pred:
                    self.y1.append(torch.Tensor([reward]))
                    self.y2.append(torch.Tensor([reward - self.f1_list[k] ]))
                else:
                    self.y1.append(torch.Tensor([1 - reward]))
                    self.y2.append(torch.Tensor([1 - reward - self.f1_list[k] ]))

        if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): 
            self.train_NN_batch(self.net1, self.X1_train, self.y1)
            self.train_NN_batch(self.net2, self.X2_train, self.y2)

        return None, None
    

    def train_NN_batch(self, model, hist_X, hist_Y, num_epochs=40, lr=0.001, batch_size=64):
        model.train()
        hist_X = torch.cat(hist_X).float()
        hist_Y = torch.cat(hist_Y).float()
        # print('hist_X', hist_X.shape, hist_Y.shape)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(hist_X, hist_Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = hist_X.size(0)

        for i in range(num_epochs):
            batch_loss = 0.0
            
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).view(-1)
                # print(pred.shape,y.shape)
                # print(pred, y)
                loss = torch.mean((pred - y) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num


