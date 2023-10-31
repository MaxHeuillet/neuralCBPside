import numpy as np
import geometry_gurobi
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy

from scipy.optimize import minimize
import copy
import pickle
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR


class OriginalNetwork(nn.Module):
    def __init__(self,  d, m):
        super(OriginalNetwork, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.activate1 = nn.Tanh() #nn.ReLU()
        self.fc2 = nn.Linear(m, m)
        self.activate2 = nn.Tanh() #nn.ReLU()
        self.fc3 = nn.Linear(m, m)
        self.activate3 = nn.Tanh() #nn.ReLU()
        self.fc4 = nn.Linear(m, 1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc4.weight, mean=0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
    def forward(self, x):
        x = self.fc4( self.activate3( self.fc3( self.activate2( self.fc2( self.activate1( self.fc1( x ) ) ) ) ) ) )
        return x

class CustomDataset(Dataset):
    def __init__(self, ):
        self.obs = None
        self.labels = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.labels[index]
    
    def append(self, X , y,):
        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0) 
        self.labels = y if self.labels is None else np.concatenate( (self.labels, y), axis=0)


class ExploreCommit():

    def __init__(self, game, budget, n_classes, lbd_neural, m  ,device):

        self.name = 'explorecommit'
        self.device = device
        self.budget = budget
        self.over_budget = False
        self.N = game.n_actions
        
        self.game = game
        self.n_classes = n_classes
        self.counter = 0
        self.H = 50
        self.m = m
        self.lbd_neural = lbd_neural

    def reset(self, d):
        self.d = d
        self.func = OriginalNetwork( self.d , self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
        self.over_budget = False
        self.counter = 0
        self.counters = [0] * self.N
            
    def load(self, path):
        original_model = OriginalNetwork(d=self.d, m=self.m)
        original_model.load_state_dict(torch.load(path))
        original_model.to(self.device)
        self.func = original_model

    def get_action(self, t, X, mode = 'train'):

        y_pred = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach().numpy()[0][0]
        self.y_pred = 1 if y_pred >= 0.5 else 2
        print('y_pred', self.y_pred, y_pred)

        if self.over_budget == False:
            action = 0
        else:
            action = self.y_pred 

        history = [action, self.counter, np.nan, self.over_budget]
    
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if self.counter>self.budget:
            self.over_budget = True

        if action == 0:
            self.counter += 1
            self.hist.append( X , [outcome] )

        global_loss = []
        if (t>self.n_classes) and (self.counter % self.H == 0):  

            self.func = copy.deepcopy(self.func0)
            optimizer = optim.Adam(self.func.parameters(), lr=0.1, weight_decay = self.lbd_neural )
            dataloader = DataLoader(self.hist, batch_size=len(self.hist), shuffle=True) 
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

            for _ in range(1000): 
                
                train_loss = self.step(dataloader, optimizer)
                current_lr = optimizer.param_groups[0]['lr']
                global_loss.append(train_loss)

                if _ % 10 == 0 :
                    scheduler.step()

                if _ % 25 == 0:
                    print('train loss', train_loss,  )

        return global_loss, None
                

    def step(self, loader, opt):
        #""Standard training/evaluation epoch over the dataset"""

        loss = 0
        for X, y in loader:
            X, y  = X.to(self.device).float(), y.to(self.device).float()
            y = y.unsqueeze(1)
            # print(X.shape, y.shape)
            pred = self.func(X)
            l = nn.MSELoss()(pred, y)
            loss += l
            opt.zero_grad()
            l.backward()
            opt.step()

        return loss.item()

