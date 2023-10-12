import numpy as np
import geometry_v3


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
from scipy.special import logit, expit
import random


class DeployedNetwork(nn.Module):
    def __init__(self,  d, m):
        super(DeployedNetwork, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(m, m)
        self.activate2 = nn.ReLU()
        self.fc3 = nn.Linear(m, m)
        self.activate3 = nn.ReLU()
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



class Egreedy():

    def __init__(self, game, m, device):

        self.name = 'egreedy'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.m = m
        self.H = 50



    def reset(self, d):
        self.d = d

        self.func = DeployedNetwork( self.d , self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()


    def get_action(self, t, X):

        prediction = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach()
        probability = expit(prediction)
        self.pred_action = 1 if probability < 0.5 else 2

        print('prediction', prediction, probability, self.pred_action)

        if random.random() < 0.1:
            action = 0
        else:
            action = self.pred_action

        explored = 1 if action ==0 else 0

        history = {'monitor_action':action, 'explore':explored,}
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if action == 0:
            self.hist.append( X , [outcome] )
            
        global_loss = []
        global_losses = []
        if (t>self.N):
            if (t % 50 == 0 and t<1000) or (t % 500 == 0 and t>=1000):

                self.func = copy.deepcopy(self.func0)
                optimizer = optim.Adam(self.func.parameters(), lr=0.1, weight_decay = 0 )
                dataloader = DataLoader(self.hist, batch_size=len(self.hist), shuffle=True) 
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

                for _ in range(1000): 
                        
                    train_loss, losses = self.step(dataloader, optimizer)
                    current_lr = optimizer.param_groups[0]['lr']
                    global_loss.append(train_loss)
                    global_losses.append(losses)
                    if _ % 10 == 0 :
                        scheduler.step()
                    # scheduler.step()
                    if _ % 25 == 0:
                        print('train loss', train_loss, 'losses', losses )

        return global_loss, global_losses
                

    def step(self, loader, opt):
        #""Standard training/evaluation epoch over the dataset"""

        for X, y in loader:
            X, y  = X.to(self.device).float(), y.to(self.device).float()
            loss = 0
            losses = []
            losses_vec =[]
 

            pred = self.func(X).squeeze(1)
            # print(pred.shape, y.shape)
            l = nn.BCEWithLogitsLoss()(pred, y)
            loss += l
            losses.append( l )
            losses_vec.append(l.item())

            opt.zero_grad()
            l.backward()
            opt.step()
            # print(losses)
        return loss.item(), losses_vec


# class Random():

#     def __init__(self, game,):
#         self.name = 'random'
#         self.game = game
#         self.N = game.n_actions

#     def get_action(self, t, context = None ):
        
#         pbt = np.ones( self.game.n_actions ) / self.game.n_actions
#         action = np.random.choice(self.game.n_actions, 1,  p = pbt )[0]
#         explored = 1 if action == 0 else 0
#         history = {'monitor_action':action, 'explore':explored,}
            
#         return action, history

#     def update(self, action, feedback, outcome, context, t):
#         return None, None

#     def reset(self, d):
#         pass
