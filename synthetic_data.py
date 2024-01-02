
import numpy as np
from numpy.linalg import norm
# from torchvision import datasets, transforms
# import torch
import numpy as np
# from mnist_c import corruptions
from scipy.stats import truncnorm
from scipy.special import expit
from scipy.special import logit, expit

import torch
from torchvision import datasets, transforms

import pickle as pkl
import gzip




import pandas as pd
import pickle
# import arff
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

class Bandit_multi:
    def __init__(self, name):
        # Fetch data
        if name == 'covertype':
            with open('./data/covertype.pkl', 'rb') as file:
                X, y = pkl.load(file)
            
            X = pd.get_dummies(X)
            # print(X,y)
            # class: 1-7
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
            unique_values = set(y.values)
            label_map = {value:i for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            
        elif name == 'MagicTelescope':
            with open('./data/MagicTelescope.pkl', 'rb') as file:
                X, y = pkl.load(file)

            # X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # class: h, g
            # avoid nan, set nan as -1
            # print(X,y)
            unique_values = set(y.values)
            label_map = {value:i for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            X[np.isnan(X)] = - 1
            X = normalize(X)

        elif name == 'shuttle':
            with open('./data/shuttle.pkl', 'rb') as file:
                X, y = pkl.load(file)
            
            # avoid nan, set nan as -1
            # print(X,y)
            X[np.isnan(X)] = - 1
            X = normalize(X)
            unique_values = set(y.values)
            label_map = {value:i for i,value in enumerate(unique_values)}
            y = y.map(label_map)
        
        elif name == 'adult':
            
            with open('./data/adult.pkl', 'rb') as file:
                X, y = pkl.load(file)

            X = pd.get_dummies(X)
            unique_values = set(y.values)
            label_map = {value:i for i,value in enumerate(unique_values)}
            y = y.map(label_map)
            X[np.isnan(X)] = - 1
            X = normalize(X)
        # elif name == 'mushroom':
        #     X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        #     # print(X,y,X.info())
        #     X = pd.get_dummies(X)
        #     unique_values = set(y.values)
        #     label_map = {value:i+1 for i,value in enumerate(unique_values)}
        #     y = y.map(label_map)
        #     # avoid nan, set nan as -1
        #     X[np.isnan(X)] = - 1
        #     X = normalize(X)
        # elif name == 'fashion':
        #     X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
        #     # print(X,y,X.info())
        #     # avoid nan, set nan as -1
        #     X[np.isnan(X)] = - 1
        #     X = normalize(X)
        # elif name == 'phishing':
        #     file_path = './binary_data/{}.txt'.format(name)
        #     f = open(file_path, "r").readlines()
        #     n = len(f)
        #     m = 68
        #     X = np.zeros([n, 68])
        #     y = np.zeros([n])
        #     for i, line in enumerate(f):
        #         line = line.strip().split()
        #         lbl = int(line[0])
        #         if lbl != 0 and lbl != 1:
        #             raise ValueError
        #         y[i] = lbl
        #         l = len(line)
        #         for item in range(1, l):
        #             pos, value = line[item].split(':')
        #             pos, value = int(pos), float(value)
        #             X[i, pos-1] = value
        # elif name == "letter":
        #     file_path = './dataset/binary_data/{}_binary_data.pt'.format(name)
        #     f = open(file_path, 'rb')
        #     data = pickle.load(f)
        #     X, y = data['X'], data['Y']   
        else:
            raise RuntimeError('Dataset does not exist')
        # Shuffle data
        self.X, self.y = shuffle(X, y)

from sklearn.model_selection import train_test_split

class CustomContexts():
    
    def __init__(self, eval):
        self.eval = eval

    def initiate_loader(self,X, y):
        self.nb_classes = len(set(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=self.eval.env_random_state)


        self.train_loader = list( zip(X_train, y_train) ) 
        self.eval.env_random_state.shuffle(self.train_loader)

        self.X_test = X_test
        self.y_test = [int(item) for item in y_test.tolist()]
    
        self.index = 0
        x, y = self.train_loader[self.index]
        print(x.shape)
        self.d = x.shape[0]
        self.len = len(self.train_loader)

    def get_context(self):
        if self.index >= self.len:
            raise IndexError("Index out of range. No more data to retrieve.")
        
        x, y = self.train_loader[self.index]
        x = x.reshape(1, -1)
        x = torch.Tensor(x).unsqueeze(0)


        val = [0] * self.nb_classes
        val[y] = 1

        self.index += 1

        return x.reshape(1, -1), val
    
    def get_test_data(self,):

        return torch.Tensor(self.X_test), self.y_test
    
####################################################################################################
####################################################################################################
#####################################################################################################

class FashionMNISTContexts():

    def __init__(self, eval):
        self.eval = eval

    def initiate_loader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        self.train_loader = [(img, label) for img, label in train_dataset]
        self.eval.env_random_state.shuffle(self.train_loader)

        test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        self.test_dataset = [(img, label) for img, label in test_dataset]
        self.eval.env_random_state.shuffle(self.test_dataset)

        self.index = 0
        x, y = self.train_loader[self.index]
        print(y)
        if self.eval.model == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.eval.model == 'LeNet':
            self.d = x.shape

    def get_context(self):
        x, y = self.test_loader[self.index]

        if self.eval.model == 'MLP':
            x = x.view(-1)  # Flatten the image

        x = x.unsqueeze(0)

        val = [0] * 10
        val[y] = 1

        self.index += 1

        return x, val
    
    def get_test_data(self,):

        # Initialize lists to store the separated features and labels
        X = []
        y = []

        # Iterate over the dataset and separate the features and labels
        for data, target in self.test_dataset:
            X.append( data.flatten().unsqueeze(0) )
            y.append(target)

        X = torch.cat(X).float().to('cuda:0')
        
        return X, y


class CIFAR10Contexts():

    def __init__(self, eval):
        self.eval = eval

    def initiate_loader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        self.train_loader = [(img, label) for img, label in train_dataset]
        self.eval.env_random_state.shuffle(self.train_loader)

        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        self.test_dataset = [(img, label) for img, label in test_dataset]
        self.eval.env_random_state.shuffle(self.test_dataset)

        self.index = 0
        x, y = self.train_loader[self.index]
        print(y)
        if self.eval.model == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.eval.model == 'LeNet':
            self.d = x.shape

    def get_context(self):
        x, y = self.test_loader[self.index]

        if self.eval.model == 'MLP':
            x = x.view(-1)  # Flatten the image

        x = x.unsqueeze(0)

        val = [0] * 10
        val[y] = 1

        self.index += 1

        return x, val
    
    def get_test_data(self,):

        # Initialize lists to store the separated features and labels
        X = []
        y = []

        # Iterate over the dataset and separate the features and labels
        for data, target in self.test_dataset:
            X.append( data.flatten().unsqueeze(0) )
            y.append(target)

        X = torch.cat(X).float().to('cuda:0')
        
        return X, y


class MNISTcontexts():

    def __init__(self, eval):
        self.eval = eval

    def initiate_loader(self,):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        self.train_loader = [(img, label) for img, label in train_dataset]
        self.eval.env_random_state.shuffle(self.train_loader)

        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        self.test_dataset = [(img, label) for img, label in test_dataset]
        self.eval.env_random_state.shuffle(self.test_dataset)

        self.index = 0
        x, y = self.train_loader[self.index]
        print(y)
        if self.eval.model == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.eval.model == 'LeNet':
            self.d = x.shape

    def get_context(self,):
        
        x, y = self.test_loader[self.index]

        if self.eval.model == 'MLP':
            x = x.flatten()
        
        x = x.unsqueeze(0)
            
        val = [0] * 10
        val[ y ] = 1
        
        self.index += 1

        return x , val 
    
    def get_test_data(self,):

        # Initialize lists to store the separated features and labels
        X = []
        y = []

        # Iterate over the dataset and separate the features and labels
        for data, target in self.test_dataset:
            X.append( data.flatten().unsqueeze(0) )
            y.append(target)

        X = torch.cat(X).float().to('cuda:0')
        
        return X, y


class MNISTcontexts_binary():

    def __init__(self, eval):
        self.eval = eval

    def initiate_loader(self,):

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        self.train_loader = [(img, label) for img, label in train_dataset]
        self.eval.env_random_state.shuffle(self.train_loader)

        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        self.test_dataset = [(img, label) for img, label in test_dataset]
        self.eval.env_random_state.shuffle(self.test_dataset)

        self.index = 0
        x, y = self.train_loader[self.index]
        print(y)
        if self.eval.model == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.eval.model == 'LeNet':
            self.d = x.shape

    def get_context(self,):
        
        x, y = self.train_loader[self.index]

        if self.eval.model == 'MLP':
            x = x.flatten()
        x = x.unsqueeze(0)

        p = 1 if y % 2 == 0 else 0
        val = [ p, 1-p ]
        self.index += 1

        return x, val 
    
    def get_test_data(self,):

        # Initialize lists to store the separated features and labels
        X = []
        y = []

        # Iterate over the dataset and separate the features and labels
        for data, target in self.test_dataset:
            X.append( data.flatten().unsqueeze(0) )
            target = 1 if target % 2 == 0 else 0
            y.append(target)

        X = torch.cat(X).float().to('cuda:0')
        
        return X, y
