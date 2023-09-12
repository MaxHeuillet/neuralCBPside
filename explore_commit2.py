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
        self.counters = np.zeros(self.n_classes)

    def reset(self, d):
        self.d = d
        self.hist = CustomDataset()
        self.over_budget = False
        self.counter = 0
        self.counters = np.zeros(self.n_classes)
            
    def get_action(self, t, X, y_pred):
        
        self.y_pred = y_pred

        if self.over_budget == False and self.counters[y_pred]<=self.budget/self.n_classes:
            action = 0
        else:
            action = 1

        history = [action, self.counter, self.y_pred, self.over_budget]
    
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if action == 0:
            self.counters[self.y_pred] += 1
            self.counter+=1
            self.hist.append( X , [outcome] )

        if (self.counters>=(self.budget/self.n_classes)).all():
            self.over_budget = True

        return None, None
                
