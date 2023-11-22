
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



class MNISTcontexts():

    def __init__(self, type):
        self.type = type

    def initiate_loader(self,):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        #train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
        self.test_loader = list(test_loader)
        self.index = 0
        x, y = self.test_loader[self.index]
        if self.type == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.type == 'LeNet':
            self.d = x.shape

    def get_context(self,):
        
        x, y = self.test_loader[self.index]
        if self.type == 'MLP':
            x = x.flatten()
            
        sample = x.numpy() #.reshape(1, -1)

        val = [0] * 10
        val[ y.item() ] = 1
        
        self.index += 1

        return sample , val 


class MNISTcontexts_binary():

    def __init__(self, type):
        self.type = type

    def initiate_loader(self,):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])
        #train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
        self.test_loader = list(test_loader)
        self.index = 0
        x, y = self.test_loader[self.index]
        if self.type == 'MLP':
            x = x.flatten()
            self.d = x.shape[0]
        elif self.type == 'LeNet':
            self.d = x.shape



    def get_context(self,):
        
        x, y = self.test_loader[self.index]

        if self.type == 'MLP':

            x = x.flatten()


        sample = x.numpy()

        p = 1 if y.item() % 2 == 0 else 0
        
        val = [ p, 1-p ]

        self.index += 1

        return sample , val 





class QuadraticContexts:

    def __init__(self):
        self.type = 'quadratic'
        self.d = 3
        self.mean = None
        self.std = None

    def set_b(self, b):
        self.b = b

    def normalization(self,):
        all = []
        for _ in range(10000):
            c, d = self.get_context(False)
            all.append(c)
        all = np.array(all)
        self.mean = np.mean(all,0)
        self.std = np.std(all,0)     
        
    def get_context(self, normalize):

        sample = np.random.uniform(-1, 1, 3)
        x,y,z = sample 
        
        decision_boundary = self.project_to_3d(x,y) - z

        p = 1 if decision_boundary >= 0 else 0

        sample = sample.reshape(1, len(sample) )
        val = [ p, 1-p ]
        if normalize:
            sample = ( sample - self.mean ) / self.std

        return sample , val 

    def project_to_3d(self, x, y):
        z = x**2 - self.b * y**2
        return z
    
    def denormalize(self,x):
        return (x+self.mean) * self.std 



class QuinticContexts:

    def __init__(self):
        self.type = 'quintic'
        self.d = 2
        self.mean = None
        self.std = None

    def set_b(self, b):
        self.b = b

    def normalization(self,):
        all = []
        for _ in range(100000):
            c, _ = self.get_context(False)
            all.append(c)
        all = np.array(all)
        self.mean = np.mean(all,0)
        self.std = np.std(all,0)   
        
    def get_context(self, normalize):

        sample = np.random.uniform(-1, 1, 2)
        x, y = sample 
        x = x + self.b
        decision_boundary = x**5 - y**5 + y**3 
        p = 1 if decision_boundary >= 0 else 0
        # p = logit(decision_boundary)

        # sample = np.array([x/2, x/2, y/3, y/3, y/3])
        # sample = sample.reshape(1, len(sample) )
        sample = sample.reshape(1, self.d)
        val = [ p, 1-p ]

        if normalize:
            sample = ( sample - self.mean ) / self.std

        return sample , val 
    
    def decision_boundary_function(self, x, y, b=0):
        x = x + b
        decision_boundary = x**5 - y**5 + y**3
        return decision_boundary >= 0
    
    def denormalize(self,x):
        return (x+self.mean) * self.std 







class QuinticContexts_imbalanced:

    def __init__(self):
        self.type = 'quintic'
        self.d = 2
        self.mean = None
        self.std = None

    def set_b(self, b):
        self.b = b

    def normalization(self,):
        all = []
        for _ in range(100000):
            c, _ = self.get_context(False)
            all.append(c)
        all = np.array(all)
        self.mean = np.mean(all,0)
        self.std = np.std(all,0)   
        
    def get_context(self, normalize):

        sample = np.random.uniform(-1, 1, 2) if np.random.uniform(0,1)<0.05 else np.random.uniform(-0.25, 0.25, 2)
        x, y = sample 
        x = x + self.b
        decision_boundary = x**5 - y**5 + y**3 
        p = 1 if decision_boundary >= 0 else 0
        # p = logit(decision_boundary)

        # sample = np.array([x/2, x/2, y/3, y/3, y/3])
        # sample = sample.reshape(1, len(sample) )
        sample = sample.reshape(1, self.d)
        val = [ p, 1-p ]

        if normalize:
            sample = ( sample - self.mean ) / self.std

        return sample , val 
    
    def decision_boundary_function(self, x, y, b=0):
        x = x + b
        decision_boundary = x**5 - y**5 + y**3
        return decision_boundary >= 0
    
    def denormalize(self,x):
        return (x+self.mean) * self.std 













# class BullsEyeContexts:
#     def __init__(self,  ):
#         self.type = 'bullseye'
#         self.d = 2
#         self.inner_radius1 = 0.65
#         self.outer_radius1 = 0.75 #0.95 
#         self.inner_radius2 = 0.15
#         self.outer_radius2 = 0.25 #0.4
#         self.mean = np.array([[0.00031191, 0.00048158]])
#         self.std = np.array([[0.57730562, 0.57743101]])

#     def get_context(self, ):

#         sample = np.random.uniform(-1, 1, 2)
#         x,y = sample 
#         distance = np.sqrt(x**2 + y**2)

#         if ( self.inner_radius1 <= distance <= self.outer_radius1) or (self.inner_radius2 <= distance <= self.outer_radius2):
#             p = 1
#         else:
#             p = 0

#         sample = sample.reshape(1, len(sample))
#         val = [ p, 1-p ]

#         # mean = np.array([[-1.80108367e-04, -1.80108367e-04, -9.08900288e-05, -9.08900288e-05, -9.08900288e-05]])
#         # std = np.array([[0.28863607, 0.28863607, 0.19246712, 0.19246712, 0.19246712]])
#         sample = ( sample - self.mean ) / self.std

#         return sample , val 
    
#     def decision_boundary_function(self, x, y):
#             distance = np.sqrt(x**2 + y**2)
#             condition1 = np.logical_and(self.inner_radius1 <= distance, distance <= self.outer_radius1)
#             condition2 = np.logical_and(self.inner_radius2 <= distance, distance <= self.outer_radius2)
#             return np.logical_or(condition1, condition2)
    
#     def denormalize(self,x):
#             return (x+self.mean) * self.std 
    
# class MixtureContexts:
#     def __init__(self,  ):
#         self.d = 5 # number of features
#         self.type = 'mixture'
#         self.num_circles = 8
#         # Generate the positions of the centers of the smaller circles
#         self.center_radius = 0.75  # Radius of the larger circle
#         self.center_angles = np.linspace(0, 2*np.pi, self.num_circles, endpoint=False)
#         self.center_x = self.center_radius * np.cos(self.center_angles)
#         self.center_y = self.center_radius * np.sin(self.center_angles)

#     def get_context(self, ):

#         sample = np.random.uniform(-1, 1, 2)
#         x,y = sample 

#         circle = False
#         # Check if the sample belongs to one of the circles
#         for i in range(self.num_circles):
#             distance = np.sqrt((x - self.center_x[i])**2 + (y - self.center_y[i])**2)
#             if distance <= 0.1:  # Adjust the distance threshold as per your preference
#                 circle = True
#                 p = 1
#                 break
#         if circle==False:
#             p = 0

#         sample = np.array([x/2, x/2, y/3, y/3, y/3])
#         sample = sample.reshape(1, self.d)
#         val = [ p, 1-p ]

#         # sample = np.array(sample)
#         # mean = np.array([0.00052123, 0.00041331])
#         # std = np.array([0.57674398, 0.57732179])
#         # sample = ( sample - mean ) / std

#         return sample , val 

# ##############################################################################
# ##############################################################################

# def truncated_gaussian(mean, variance, a, b, size):
#     # Calculate the standard deviation from the variance
#     std_dev = np.sqrt(variance)

#     # Calculate the lower and upper bounds for truncation
#     lower_bound = (a - mean) / std_dev
#     upper_bound = (b - mean) / std_dev

#     # Generate samples from the truncated normal distribution
#     samples = truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)

#     return np.array(samples)

# class LinearContexts:
#     def __init__(self,  d, task):
#         self.d = d
#         self.w = np.ones(self.d) * 1/(2*self.d)
#         self.b = 1/2
#         self.type = 'linear'
#         self.task = task
    
#     def get_context(self, ):

#         if self.task == 'imbalanced':
#             p = np.random.uniform(0, 0.2) if np.random.uniform(0, 1)<0.5 else np.random.uniform(0.8, 1)
#             mean = np.array([-0.00069175, -0.00069175, -0.00069175, -0.00069175, -0.00069175])
#             std = np.array([4.04096508, 4.04096508, 4.04096508, 4.04096508, 4.04096508])
#         elif self.task == 'balanced':
#             p = np.random.uniform(0.4, 0.6) 
#             mean = np.array([-0.00060318, -0.00060318, -0.00060318, -0.00060318, -0.00060318])
#             std = np.array([0.57770456, 0.57770456, 0.57770456, 0.57770456, 0.57770456])

#         context = (p-self.b) * 1/self.w
#         val = [ p, 1-p ]

#         context = np.array(context)
#         context = ( context - mean ) / std

#         return context , val 

# class LinearContexts:
#     def __init__(self,  w, task):
#         self.d = len(w) # number of features
#         self.w = w
#         self.type = 'linear'
#         self.task = task
    
#     def get_context(self, ):

#         if self.task == 'imbalanced':
#             context = truncated_gaussian(0, 0.1, 0, 1, self.d) if np.random.uniform(0,1)<0.5 else truncated_gaussian(1, 0.1, 0, 1, self.d)
#         elif self.task == 'balanced':
#             context = truncated_gaussian(0.5, 1, 0, 1, self.d)

#         # cont = context.reshape(self.d,1)
#         p = self.w @ context
#         val = [ p, 1-p ]

#         context = np.array(context)
#         mean = np.array([0.49894511, 0.49964278, 0.49914392, 0.50041341, 0.49964411])
#         std = np.array([0.31208973, 0.31217681, 0.31190383, 0.31202762, 0.31117992])
#         context = ( context - mean ) / std

#         return context , val 

# class QuadraticContexts:
#     def __init__(self,  w, task):
#         self.d = len(w) # number of features
#         self.w = w
#         self.type = 'quadratic'
#         self.task = task
    
#     def get_context(self, ):

#         if self.task == 'imbalanced':
#             context = truncated_gaussian(0, 0.2, 0, 1, self.d ) if np.random.uniform(0,1)<0.5 else truncated_gaussian(1, 0.025, 0,  1, self.d )
#         else:
#             context = truncated_gaussian(1/np.sqrt(2), 0.05, 0,  1, self.d )

#         cont = np.array(context)
#         val =  (self.w @ cont)**2 
#         val = [ val, 1-val ]

#         mean = np.array([0.60673928, 0.60526987, 0.60531421, 0.60608301, 0.6049472 ])
#         std = np.array([0.32465229, 0.32458575, 0.324807,   0.32434703, 0.32378323])
#         cont = ( cont - mean ) / std

#         return cont, val 

# class NonLinearContexts:
#     def __init__(self,  w, task):
#         self.d = len(w) # number of features
#         self.w = w
#         self.type = 'nonlinear'
#         self.task = task
    
#     def get_context(self, ):

#         if self.task == 'imbalanced':
#             context = np.random.uniform(-10, 10, self.d) 
#         # else:
#             # context = truncated_gaussian(0, 5, -10,  10, self.d )

#         cont = np.array(context)
#         # print(cont)

#         val =  (np.sin(  self.w @ context  ) + 1 ) / 2 
#         val = [ val, 1-val ]

#         mean = np.array([ 0.00210544, -0.02539498,  0.01848252, -0.00751817,  0.01412415])
#         std = np.array([5.77238894, 5.77531259, 5.7664777,  5.77564962, 5.77461231])
#         cont = ( cont - mean ) / std

#         return cont, val 


# class SinusoidContexts:
#     def __init__(self,  w, task):
#         self.d = len(w) # number of features
#         self.w = w
#         self.type = 'sinusoid'
#         self.task = task
    
#     def get_context(self, ):

#         if self.task == 'imbalanced':
#             if np.random.uniform(0,1)<0.725:
#                 context = truncated_gaussian(np.pi/self.d**2, 0.05, 0, np.pi, self.d) if np.random.uniform(0,1)<0.5 else  truncated_gaussian(0, 0.05, 0, np.pi, self.d)
#             else :
#                 context = truncated_gaussian(np.pi/2, self.d * np.pi, 0, np.pi, self.d)
                
#         else:
#             context = truncated_gaussian(np.pi/6, 0.1, 0, np.pi, self.d) if np.random.uniform(0,1)<0.5 else  truncated_gaussian(5 * np.pi / 6, 0.1, 0, np.pi, self.d)
        
        
#         cont = np.array(context) 
#         val = np.sin(self.w@cont)
#         val = [ val, 1-val ]

#         mean = np.array([0.5812034 , 0.5774185 , 0.57832179, 0.57731911, 0.57838398])
#         std = np.array([0.78190242, 0.77617083, 0.77710955, 0.77693384, 0.77836506])
#         cont = ( cont - mean ) / std


#         return cont, val 

# class PolynomialContexts:
#     def __init__(self, d, margin):
#         self.d = d #number of features
#         self.margin = margin #np.random.uniform(0,0.5) # decision boundary
#         self.type = 'polynomial'
#         self.d_context = 28

#     def get_context(self, ):
#         context = np.random.uniform(-1, 1,  self.d )
#         return np.array(context).reshape(2,1)
    
    # while True:
    # x,y = context
    # if   y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 - self.margin  and label == 0:
    #     return np.array(context).reshape(2,1) #/ norm(context, 1)
    # elif   y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and label == 1:
    #     return np.array(context).reshape(2,1) #/ norm(context, 1)
    # def generate_unique_context(self,):
    #     self.context_A = []
    #     self.context_B = []
    #     while  len(self.context_A) == 0 or len(self.context_B) == 0:
    #         context = np.random.uniform(-1, 1, self.d)
    #         x,y = context
    #         if y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_A) == 0 :
    #             self.context_A = np.array(context).reshape(2,1) #/ norm(context, 1)
    #         elif y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_B) == 0:
    #             self.context_B = np.array(context).reshape(2,1) #/ norm(context, 1)
    # def get_same_context(self, label):
    #     if label == 0:
    #             return self.context_A
    #     elif label == 1:
    #             return self.context_B

# class LinearContexts:
#     def __init__(self, game, w):
#         self.game = game
#         self.d = len(w) #number of features
#         self.w = w
#         self.type = 'linear'

#     def get_context(self, ):
#         context = np.random.uniform(0, 1,  self.d )
#         return np.array(context).reshape(self.d,1)
#     # def step(self): #get context
#     #     assert self.cursor < self.size
#     #     X = np.zeros((self.n_arm, self.dim))
#     #     for a in range(self.n_arm):
#     #         X[a, a * self.act_dim:a * self.act_dim + self.act_dim] = self.X[self.cursor]
#     #     arm = self.y_arm[self.cursor][0]
#     #     rwd = np.zeros((self.n_arm,))
#     #     rwd[arm] = 1
#     #     self.cursor += 1
#     #     return X, rwd
    
#     def get_distribution(self,cont):
#         val = self.w @ cont
#         return [ val[0], 1-val[0] ]
    
    # while True:
    # if   self.w.T @ context + self.b > self.margin and label == 0:
    #     return np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # elif  self.w.T @ context  + self.b < -self.margin and label == 1:
    #     return np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # def generate_unique_context(self,):
    #     self.context_A = []
    #     self.context_B = []
    #     while  len(self.context_A) == 0 or len(self.context_B) == 0:
    #         context = np.random.uniform(-1, 1, self.d)
    #         if self.w.T @ context + self.b > self.margin and len(self.context_A) == 0 :
    #             self.context_A = np.array(context).reshape(self.d,1) #/ norm(context, 1)
    #         elif self.w.T @ context + self.b < -self.margin and len(self.context_B) == 0:
    #             self.context_B = np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # def get_same_context(self, label):
    #     if label == 0:
    #             return self.context_A
    #     elif label == 1:
    #             return self.context_B

# class ToyContexts:

#     def __init__(self, ):
#         self.type = 'toy'
#         self.d_context = 2

#     def get_context(self, label):
#         while True:
#             context =   np.random.randint(2) # np.random.uniform(0, 1 )
#             if   context >= 0.5 and label == 0:
#                 return np.array([1,context]).reshape(2,1) #/ norm(context, 1)
#             elif  context < 0.5 and label == 1:
#                 return np.array([1,context-1]).reshape(2,1) #/ norm(context, 1)

#     def generate_unique_context(self,):
#         self.context_A = []
#         self.context_B = []
#         while  len(self.context_A) == 0 or len(self.context_B) == 0:
#             context =  np.random.randint(2) # np.random.uniform(0, 1) #
#             if context >= 0.5 and len(self.context_A) == 0 :
#                 self.context_A = np.array([1,context]).reshape(2,1) #/ norm(context, 1)
#             elif context < 0.5 and len(self.context_B) == 0:
#                 self.context_B = np.array([1,context-1]).reshape(2,1) #/ norm(context, 1)

#     def get_same_context(self, label):
#         if label == 0:
#                 return self.context_A
#         elif label == 1:
#                 return self.context_B
        

# class OrthogonalContexts:

#     def __init__(self, d):
#         self.type = 'orthogonal'
#         self.d = d

#     def get_context(self, label):
#         idx = np.random.randint(self.d)
#         context = np.zeros( (self.d,1) )
#         context[idx] = 1
#         return context



