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
    f2 = net2(dc)
    return f1.item(), f2.item(), dc

class INeurAL():

    def __init__(self, device, budget, d):
        self.name = 'ineural'
        
        self.device = device

        self.d = d

        self.budget = budget
        self.query_num = 0
        self.margin = 7
        self.N = 3

        self.ber = 1.1

        self.X1_train, self.X2_train, self.y1, self.y2 = [], [], [], []

        self.net1 = Network_exploitation(self.d * 2).to(self.device)
        self.net2 = Network_exploration(self.d * 2 * 2).to(self.device)

    def reset(self,):

        self.query_num = 0
        X1_train, X2_train, y1, y2 = [], [], [], []
        self.net1 = Network_exploitation(self.d * 2).to(self.device)
        self.net2 = Network_exploration(self.d * 2 * 2).to(self.device)

    def get_action(self, t, X):

        print('X shape', X.shape)
        X = torch.from_numpy(X).to(self.device)
        ci = torch.zeros(1, self.d).to(self.device)
        self.x0 = torch.cat([X, ci], dim=1).to(torch.float32)
        self.x1 = torch.cat([ci, X], dim=1).to(torch.float32)

        self.f1_0, self.f2_0, self.dc_0 = EE_forward(self.net1, self.net2, self.x0)
        self.f1_1, self.f2_1, self.dc_1 = EE_forward(self.net1, self.net2, self.x1)
        u0 = self.f1_0 + 1 / (t+1) * self.f2_0
        u1 = self.f1_1 + 1 / (t+1) * self.f2_1

        explored = 0
        if u0 > u1:
            self.pred = 2
            if u0 - u1 < self.margin * 0.1:
                explored = 1
        else:
            self.pred = 1
            if u1 - u0 < self.margin * 0.1:
                explored = 1

        if (explored ==1) and (self.query_num < self.budget):
            if random.random() > self.ber:
                explored = 1

        action = self.pred if explored == 0 else 0

        history = {'monitor_action':action, 'explore':explored, }

        return action, history
        

    def update(self, action, feedback, outcome, t, X):

        lbl = 1 if outcome == 1 else 2
        if self.pred != lbl:
            reward = 0
        else:
            reward = 1

        if (action == 0) and (self.query_num < self.budget): 
            self.query_num += 1

            if self.pred == 1:
                self.X1_train.append(self.x0)
                self.X2_train.append(self.dc_0)
                self.y1.append(torch.Tensor([reward]))
                self.y2.append(torch.Tensor([reward - self.f1_0 - self.f2_0]))

                self.X1_train.append(self.x1)
                self.X2_train.append(self.dc_1)
                self.y1.append(torch.Tensor([1 - reward]))
                self.y2.append(torch.Tensor([1 - reward - self.f1_1 - self.f2_1]))

            elif self.pred == 2:
                self.X1_train.append(self.x1)
                self.X2_train.append(self.dc_1)
                self.y1.append(torch.Tensor([reward]))
                self.y2.append(torch.Tensor([reward - self.f1_1 - self.f2_1]))

                self.X1_train.append(self.x0)
                self.X2_train.append(self.dc_0)
                self.y1.append(torch.Tensor([1 - reward]))
                self.y2.append(torch.Tensor([1 - reward - self.f1_0 - self.f2_0]))


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

        for i in range(1000):
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


    

# def test(net1, net2, X, y):
#     dataset = TensorDataset(X, y)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
#     num = X.size(0)
#     acc = 0.0
#     ci = torch.zeros(1, X.shape[1]).to(device)

#     for i in range(num):
#         x, y = dataset[i]
#         x = x.view(1, -1).to(device)
#         x0 = torch.cat([x, ci], dim=1)
#         x1 = torch.cat([ci, x], dim=1)
#         u0 = net1(x0)
#         u1 = net1(x1)
        
#         lbl = y.item()
#         if u0 > u1:
#             pred = 0
#         else:
#             pred = 1
#         if pred == lbl:
#             acc += 1
    
#     print("Test Acc:{:.2f}".format(acc * 100.0 / num))


    


# device = 'cuda'
# test_mode = 'regret'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# # [phishing, ijcnn, letter, fashion, mnist, cifar]
# dataset_name = 'phishing'
# if dataset_name in ['mnist', 'phishing']:
#     margin = 6
# else:
#     margin = 7

# print(dataset_name, label_ratio)

# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

# if __name__ == "__main__":
#     if test_mode == 'regret':
#         X, Y = get_data(dataset_name)
#     elif test_mode == 'accuracy':
#         X, Y, test_X, test_Y = get_pop_data(dataset_name)
    
#     dataset = TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.int64)))


#     # net1 = Network_exploitation(X.shape[1] * 2).to(device)
#     # net2 = Network_exploration(X.shape[1] * 2 * 2).to(device)
#     regret = []
#     # X1_train, X2_train, y1, y2 = [], [], [], []
#     n = len(dataset)
#     # budget = int(n * label_ratio)
#     current_regret = 0.0
#     # query_num = 0
#     tf = time.time()
#     # ci = torch.zeros(1, X.shape[1]).to(device)

#     # if label_ratio <= 0.1:
#     #     ber = 1.1
#     # else:
#     #     ber = get_ber(label_ratio)

#     for i in range(n):
        # x, y = dataset[i]
        # x = x.view(1, -1).to(device)
        # x0 = torch.cat([x, ci], dim=1)
        # x1 = torch.cat([ci, x], dim=1)

        # f1_0, f2_0, dc_0 = EE_forward(net1, net2, x0)
        # f1_1, f2_1, dc_1 = EE_forward(net1, net2, x1)
        # u0 = f1_0 + 1 / (i+1) * f2_0
        # u1 = f1_1 + 1 / (i+1) * f2_1

        # ind = 0
        # if u0 > u1:
        #     pred = 0
        #     if u0 - u1 < margin * 0.1:
        #         ind = 1
        # else:
        #     pred = 1
        #     if u1 - u0 < margin * 0.1:
        #         ind = 1

        # lbl = y.item()
        # if pred != lbl:
        #     current_regret += 1
        #     reward = 0
        # else:
        #     reward = 1

        # if not ind and query_num < budget:
        #     if random.random() > ber:
        #         ind = 1

        # if ind and (query_num < budget): 
        #     query_num += 1
        #     if pred == 0:
        #         X1_train.append(x0)
        #         X2_train.append(dc_0)
        #         y1.append(torch.Tensor([reward]))
        #         y2.append(torch.Tensor([reward - f1_0 - f2_0]))

        #         X1_train.append(x1)
        #         X2_train.append(dc_1)
        #         y1.append(torch.Tensor([1 - reward]))
        #         y2.append(torch.Tensor([1 - reward - f1_1 - f2_1]))
        #     else:
        #         X1_train.append(x1)
        #         X2_train.append(dc_1)
        #         y1.append(torch.Tensor([reward]))
        #         y2.append(torch.Tensor([reward - f1_1 - f2_1]))

        #         X1_train.append(x0)
        #         X2_train.append(dc_0)
        #         y1.append(torch.Tensor([1 - reward]))
        #         y2.append(torch.Tensor([1 - reward - f1_0 - f2_0]))

        #     train_NN_batch(net1, X1_train, y1)
        #     train_NN_batch(net2, X2_train, y2)

        # if (i+1) % 1000 == 0:
        #     print("Time:{:.2f}\tIters:{}\tRegret:{:.1f}".format(time.time()-tf, i+1, current_regret))
        #     tf = time.time()
    #     regret.append(current_regret)
       
    # print(query_num)
    # if test_mode == 'regret':
    #     print(current_regret)
    # else:
    #     test_X, test_Y = torch.tensor(test_X.astype(np.float32)), torch.tensor(test_Y.astype(np.int64))
    #     test(net1, net2, test_X, test_Y)