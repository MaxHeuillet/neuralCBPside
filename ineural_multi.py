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

class INeurALmulti():

    def __init__(self, device, budget, d, num_cls):
        self.name = 'ineuralmulti'
        
        self.device = device

        self.d = d
        self.num_cls = num_cls

        self.budget = budget
        self.query_num = 0
        self.margin = 7 # on MNIST they use 3
        self.N = 3

        self.ber = 1.1

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        input_dim = self.d + (self.num_cls-1) * self.d
        # print( 'input dim',  input_dim)
        self.net1 = Network_exploitation(input_dim).to(self.device)
        self.net2 = Network_exploration(input_dim * 2).to(self.device)

    def reset(self,):

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        input_dim = self.d + (self.num_cls-1) * self.d
        self.net1 = Network_exploitation(input_dim).to(self.device)
        self.net2 = Network_exploration(input_dim * 2).to(self.device)

    def get_action(self, t, X):

        print('X shape', X.shape)
        X = torch.from_numpy(X).to(self.device)
        # x = x.view(1, -1).to(device)

        ci = torch.zeros(1, self.d).to(self.device)
        
        self.x_list = []
        for k in range(self.num_cls):
            inputs = []
            for l in range(k):
                inputs.append(ci)
            inputs.append(X)
            for l in range(k+1, self.num_cls):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1).to(torch.float32)
            self.x_list.append(inputs)

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

        self.pred = self.u_list[0][0] + 1
        print('pred',self.pred)
        diff = self.u_list[0][1] - self.u_list[1][1]
        if diff < self.margin * 0.1:
            explored = 1
        else:
            explored = 0

        # if (explored ==1) and (self.query_num < self.budget):
        #     if random.random() > self.ber:
        #         explored = 1

        action = self.pred if explored == 0 else 0

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
        

    def update(self, action, feedback, outcome, t, X):

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
                self.X1_train.append( self.x_list[k].detach().cpu())
                self.X2_train.append( self.dc_list[k].detach().cpu())
                if k_prime == self.pred:
                    self.y1.append(torch.Tensor([reward]))
                    self.y2.append(torch.Tensor([reward - self.f1_list[k] - self.f2_list[k]]))
                else:
                    self.y1.append(torch.Tensor([1 - reward]))
                    self.y2.append(torch.Tensor([1 - reward - self.f1_list[k] - self.f2_list[k]]))

            self.train_NN_batch(self.net1, self.X1_train, self.y1)
            self.train_NN_batch(self.net2, self.X2_train, self.y2)

        return None, None
    

    def train_NN_batch(self, model, X, y, num_epochs=10, lr=0.001, batch_size=64):
        model.train()
        X = torch.cat(X).float()
        y = torch.cat(y).float()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = X.size(0)

        for i in range(num_epochs):
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

