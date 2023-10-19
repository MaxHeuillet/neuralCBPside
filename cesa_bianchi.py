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



class DeployedNetwork(nn.Module):
    def __init__(self,  d, m, output):
        super(DeployedNetwork, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.activate1 = nn.ReLU()
        self.fc2 = nn.Linear(m, output)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = self.fc2( self.activate1( self.fc1( x ) ) ) 
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

class CesaBianchi():

    def __init__(self, game, m, device):

        self.name = 'cesabianchi'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.m = m
        self.H = 50

        self.K = 0


    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.func = DeployedNetwork( self.d , self.m, 1).to(self.device)

        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
        self.feedbacks = []

        self.K = 0
        self.beta = 1

    def get_action(self, t, X):

        prediction = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach()

        probability = expit(prediction)
        self.pred_action = 1 if probability < 0.5 else 2

        print('prediction', prediction, self.pred_action)


        b = self.beta * np.sqrt(1+self.K) 
        
        p = b / ( b + abs( probability ) )

        self.Z = np.random.binomial(1, p)
        self.Z = 1-self.Z

        if self.Z == 1:
            action = 0
        else:
            action = self.pred_action

        explored = 1 if self.Z == 1 else 0

        history = {'monitor_action':action, 'explore':explored,}
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if self.Z == 1:
            self.hist.append( X , [outcome] )
            if (self.pred_action == 1 and outcome == 0) or (self.pred_action == 2 and outcome ==1):
                self.K += 1
            
        global_loss = []
        global_losses = []
        if (t>self.N):
            if (t % 50 == 0 and t<1000) or (t % 500 == 0 and t>=1000):

                self.func = copy.deepcopy(self.func0)
                optimizer = optim.Adam(self.func.parameters(), lr=0.001, weight_decay = 0 )
                dataloader = DataLoader(self.hist, batch_size=len(self.hist), shuffle=True) 
                #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

                loss = nn.BCEWithLogitsLoss()
 

                for _ in range(1000): 
                        
                    train_loss, losses = self.step(dataloader, loss, optimizer)
                    current_lr = optimizer.param_groups[0]['lr']
                    global_loss.append(train_loss)
                    global_losses.append(losses)
                    # if _ % 10 == 0 :
                    #     scheduler.step()
                    # scheduler.step()
                    if _ % 25 == 0:
                        print('train loss', train_loss, 'losses', losses )

        return global_loss, global_losses
                

    def step(self, loader, loss_func, opt):
        #""Standard training/evaluation epoch over the dataset"""

        for X, y in loader:
            X, y  = X.to(self.device).float(), y.to(self.device).float()

            loss = 0
            losses = []
            losses_vec =[]
 

            pred = self.func(X).squeeze(1)
            # print(pred.shape, y.shape)
            l = loss_func(pred, y)

            loss += l
            losses.append( l )
            losses_vec.append(l.item())

            opt.zero_grad()
            l.backward()
            opt.step()
            # print(losses)
        return loss.item(), losses_vec