import numpy as np
# import geometry_v3


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
        self.A = None #geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.m = m
        self.H = 50

        self.K = 0

    def predictor(self,X,y):
        y_pred = self.func(X).cpu().detach()
        y_proba = expit(y_pred)
        transformed_probas = torch.cat((1-y_proba, y_proba), dim=1)
        return transformed_probas


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
        self.norm_hist = 0

    def get_action(self, t, X):
        print(' ')

        X_norm = X.float().to(self.device) / np.linalg.norm( X.detach().cpu() )

        prediction = self.func( X.float().to(self.device) ).cpu().detach()

        norm = np.linalg.norm( X_norm.detach().cpu() )
        print('norm hist', self.norm_hist, 'current norm', norm)

        self.X_prime = max( self.norm_hist, norm  )

        probability = expit( prediction.item() )
        self.pred_action = 1 if probability < 0.5 else 2

        print('prediction', prediction, 'proba', probability, 'prediction', self.pred_action)


        b = self.beta * np.sqrt(self.K+1) * self.X_prime**2
        
        p = b / ( b + abs( probability ) )
        print('b', b, 'probability', p)

        self.Z = np.random.binomial(1, p)
        self.Z = 1-self.Z

        if self.Z == 1:
            action = 0
        else:
            action = self.pred_action

        explored = 1 if self.Z == 1 else 0

        if t<self.N:
            action = t

        history = {'monitor_action':action, 'explore':explored,}
            
        return action, history

    def update(self, action, feedback, outcome, t, X, loss):

        if action == 0:
            self.hist.append( X , [outcome] )
            if (self.pred_action == 1 and outcome == 0) or (self.pred_action == 2 and outcome ==1):
                self.K += 1
                self.norm_hist = self.X_prime

        # if (t>self.N):
        #     if (t<=50) or (t % 50 == 0 and t<1000 and t>50) or (t % 500 == 0 and t>=1000): #
            
        if action == 0 and (t>self.N):
            losses = self.step(self.func, self.hist)

        return None, None
                

    def step(self, model, data, num_epochs=40, lr=0.001, batch_size=64):
        #""Standard training/evaluation epoch over the dataset"""
        dataloader = DataLoader(data, batch_size=len(self.hist), shuffle=True) 
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss = nn.BCEWithLogitsLoss()
        num = len(self.hist)

        for _ in range(40):
            batch_loss = 0.0

            for X, y in dataloader:
                X, y  = X.to(self.device).float(), y.to(self.device).float()


                pred = self.func(X).squeeze(1)
                # print(pred.shape, y.shape)
                l = loss(pred, y)

                batch_loss += l.item()


                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # print(losses)

            if batch_loss / num <= 1e-3:
                return batch_loss / num
                
        return None