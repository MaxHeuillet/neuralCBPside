import numpy as np

import torch
import torch.nn as nn

from skimage.measure import block_reduce

############################################################ ############################################################ 
############################################################  MLP
############################################################ ############################################################ 

class Network_exploitation_MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=100):
        super(Network_exploitation_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        intermediate = self.activate(x)
        final_output = self.fc2(intermediate)
        return final_output, intermediate 
    

############################################################ ############################################################ 
############################################################  LeNet
############################################################ ############################################################ 

from skimage.measure import block_reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network_exploitation_LeNet(nn.Module):
    def __init__(self, output_dim):
        super(Network_exploitation_LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # Apply the layers in forward pass
        x = self.activate(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.activate(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16*5*5)  # Flatten the tensor
        
        latent = self.activate(self.fc1(x))
        x = self.activate(self.fc2(latent))
        x = self.fc3(x)
        return x, latent
    

############################################################ ############################################################ 
############################################################  Common
############################################################ ############################################################ 
    
class Network_exploration(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=100):
        super(Network_exploration, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x, None

def EE_forward(net1, net2, x):

    x.requires_grad = True
    f1, latent = net1(x)
    net1.zero_grad()

    f1.sum().backward(retain_graph=True)
    dc = torch.cat([p.grad.flatten().detach() for p in net1.parameters()])
    dc = block_reduce(dc.cpu(), block_size=51, func=np.mean)
    dc = torch.from_numpy(dc).to('cuda:0')
    
    dc = torch.cat([ dc.detach(), latent[0].detach() ]  ) 
    dc = dc / torch.linalg.norm(dc)

    f2, _ = net2(dc)

    return f1, f2, dc.unsqueeze(0)

