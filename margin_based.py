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
        self.feedbacks = None
        self.actions = None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        return self.obs[index], self.labels[index], self.feedbacks[index], self.actions[index]
    
    def append(self, X , y, f, a):
        self.obs = X if self.obs is None else np.concatenate( (self.obs, X), axis=0) 
        self.labels = y if self.labels is None else np.concatenate( (self.labels, y), axis=0)
        self.feedbacks = [[f]] if self.feedbacks is None else np.concatenate( (self.feedbacks, [[f]] ), axis=0)
        self.actions = [[a]] if self.actions is None else np.concatenate( (self.actions, [[a]] ), axis=0)

class MarginBased():

    def __init__(self, game, budget, m, device):

        self.name = 'helmbolt'
        self.device = device

        self.game = game

        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.budget = budget
        self.counter = 0
        self.over_budget = False

        self.m = m
        self.H = 50

        self.K = 0


    def reset(self, d):
        self.d = d
        
        self.memory_pareto = {}
        self.memory_neighbors = {}

        self.func = DeployedNetwork( self.d , self.m).to(self.device)
        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
        self.feedbacks = []

        self.over_budget = False
        self.counter = 0

        self.K = 0
        self.beta = 1

    def get_action(self, t, X, prediction):

        prediction = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach()
        probability = torch.softmax(prediction, dim=0).numpy()[0][0]
        pred_action = 0 if probability < 0.5 else 1

        print('prediction', prediction, probability, pred_action)

        b = self.beta * np.sqrt(1+self.K) 
        
        p = b / ( b + abs( probability ) )

        Z = np.random.binomial(1, p)

        if Z == 1 and self.over_budget==False:
            action = 0
        else:
            action = pred_action

        history = [ action, prediction, self.over_budget ]
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if self.counter > self.budget:
            self.over_budget = True

        if action == 0:
            
            self.counter += 1

            if outcome == 0:
                self.K += 1
            

        ### update exploration component:
        e_y = np.zeros( (self.M,1) )
        e_y[outcome] = 1
        Y_t = self.game.SignalMatrices[action] @ e_y 

        # print('action', action, 'feedback', feedback, 'Y_t', Y_t, 'latentX', self.latent_X)

        # print('weights', weights.shape, 'Y_t', Y_t.shape, )
        self.hist.append( X , Y_t, feedback, action )
        global_loss = []
        global_losses = []
        if (t>self.N) and (t % self.H == 0):  

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

        # symbols = [ np.unique(self.game.FeedbackMatrix[i,...]) for i in range(self.N) ]
        symbols = [ [0] ]

        for X, y, feedbacks, actions in loader:
            X, y  = X.to(self.device).float(), y.to(self.device).float()
            fdks = torch.nn.functional.one_hot(feedbacks[:,0], num_classes= self.A).to(self.device).float()
            loss = 0
            losses = []
            losses_vec =[]
            for i in [0]:  
                mask = (actions == i)[:,0]
                X_filtered = X[mask]
                fdks_filtered = fdks[mask]
                for s in symbols[i]:
                    y_filtered = fdks_filtered[:,s].unsqueeze(1)
                    pred = self.func(X_filtered)
                    l = nn.BCEWithLogitsLoss()(pred, y_filtered)
                    loss += l
                    losses.append( l )
                    losses_vec.append(l.item())
            # Stack the loss elements into a tensor
            # print('losses before', losses)
            loss_tensor = torch.stack(losses)
            # print('losses after', loss_tensor)
            loss_sum = torch.sum(loss_tensor)
            # print('losses sum', loss_sum)
            # ch.tensor(losses).to(self.device)
            # print(loss_sum )
            opt.zero_grad()
            loss_sum.backward()
            opt.step()
            # print(losses)
        return loss.item(), losses_vec