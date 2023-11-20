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

# from utils import get_data
# from load_data import load_mnist_1d
from skimage.measure import block_reduce
# from load_data_addon import Bandit_multi



class Network_exploitation(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploitation, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))
    
    
class Network_exploration(nn.Module):
    def __init__(self, dim, hidden_size=100, k=10):
        super(Network_exploration, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, k)

    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1 = net1(x)
    net1.zero_grad()
    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    #dc = dc / torch.linalg.norm(dc)
    dc = block_reduce(dc.cpu(), block_size=51, func=np.mean)
    dc = torch.from_numpy(dc).to(x.device)
    print('dc shape', dc.shape)
    f2 = net2(dc)
    return f1, f2, dc


class NeuronAL():

    def __init__(self, budget, num_cls, device):
        self.name = 'neuronal'
        
        self.device = device

        self.num_cls = num_cls

        self.budget = budget
        self.query_num = 0
        self.margin = 3 #according to their parameter search
        self.N = num_cls+1
        self.ber = 1.1

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

    def reset(self, d):

        self.d = d

        self.query_num = 0
        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        explore_size = 1560 if self.num_cls==10 else 1544 
        self.net1 = Network_exploitation(self.d, k=self.num_cls).to(self.device)
        self.net2 =Network_exploration(explore_size, k=self.num_cls).to(self.device)


    def get_action(self, t, X):

        print('X shape', X.shape)
        
        self.X = torch.from_numpy(X).to(self.device)
        self.f1, self.f2, self.dc = EE_forward(self.net1, self.net2, self.X)
        u = self.f1[0] + 1 / (self.query_num+1) * self.f2
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

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
    
    def update(self, action, feedback, outcome, t, X):

        lbl = outcome #+1 

        if (action == 0) and (self.query_num < self.budget): 
            self.query_num += 1

            self.X1_train.append(self.X)
            self.X2_train.append(torch.reshape(self.dc, (1, len(self.dc))))
            r_1 = torch.zeros(self.num_cls).to(self.device)
            r_1[lbl] = 1
            self.y1.append(r_1) 
            self.y2.append((r_1 - self.f1)[0])
            
        if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): #
            # print('X1_train',self.X1_train)
            self.train_NN_batch(self.net1, self.X1_train, self.y1)
            self.train_NN_batch(self.net2, self.X2_train, self.y2)

        return None, None
        

    def train_NN_batch(self, model, X, Y, num_epochs=10, lr=0.001, batch_size=64):
        model.train()
        X = torch.cat(X).float()
        Y = torch.stack(Y).float().detach()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num = X.size(1)

        for i in range(num_epochs):
            batch_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = torch.reshape(y, (1,-1))
                pred = model(x).view(-1)

                optimizer.zero_grad()
                loss = torch.mean((pred - y) ** 2)
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
            
            if batch_loss / num <= 1e-3:
                return batch_loss / num

        return batch_loss / num
        