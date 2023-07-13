
import numpy as np
from numpy.linalg import norm
# from torchvision import datasets, transforms
# import torch
import numpy as np
# from mnist_c import corruptions
from scipy.stats import truncnorm
from scipy.special import expit


#############################################################################
#############################################################################


class BullsEyeContexts:
    def __init__(self,  ):
        self.d = 5 # number of features
        self.type = 'bullseye'
        self.inner_radius1 = 0.7
        self.outer_radius1 = 0.8
        self.inner_radius2 = 0.2
        self.outer_radius2 = 0.3

    def get_context(self, ):

        sample = np.random.uniform(-1, 1, 2)
        x,y = sample 
        distance = np.sqrt(x**2 + y**2)

        if ( self.inner_radius1 <= distance <= self.outer_radius1) or (self.inner_radius2 <= distance <= self.outer_radius2):
            p = 1
        else:
            p = 0

        sample = np.array([x/2, x/2, y/3, y/3, y/3])
        sample = sample.reshape(1, self.d)
        val = [ p, 1-p ]

        mean = np.array([[-0.00057691, -0.00057691, -0.00016921, -0.00016921, -0.00016921]])
        std = np.array([[0.28898979, 0.28898979, 0.1923455 , 0.1923455 , 0.1923455 ]])
        sample = ( sample - mean ) / std

        return sample , val 
    

class MixtureContexts:
    def __init__(self,  ):
        self.d = 5 # number of features
        self.type = 'mixture'
        self.num_circles = 8
        # Generate the positions of the centers of the smaller circles
        self.center_radius = 0.75  # Radius of the larger circle
        self.center_angles = np.linspace(0, 2*np.pi, self.num_circles, endpoint=False)
        self.center_x = self.center_radius * np.cos(self.center_angles)
        self.center_y = self.center_radius * np.sin(self.center_angles)

    def get_context(self, ):

        sample = np.random.uniform(-1, 1, 2)
        x,y = sample 

        circle = False
        # Check if the sample belongs to one of the circles
        for i in range(self.num_circles):
            distance = np.sqrt((x - self.center_x[i])**2 + (y - self.center_y[i])**2)
            if distance <= 0.1:  # Adjust the distance threshold as per your preference
                circle = True
                p = 1
                break
        if circle==False:
            p = 0

        sample = np.array([x/2, x/2, y/3, y/3, y/3])
        sample = sample.reshape(1, self.d)
        val = [ p, 1-p ]

        # sample = np.array(sample)
        # mean = np.array([0.00052123, 0.00041331])
        # std = np.array([0.57674398, 0.57732179])
        # sample = ( sample - mean ) / std

        return sample , val 
    

class QuinticContexts:
    def __init__(self,  ):
        self.d = 5 # number of features
        self.type = 'quintic'
        
    def get_context(self, ):

        sample = np.random.uniform(-1, 1, 2)
        x,y = sample 

        # Evaluate the quintic function as the decision boundary
        decision_boundary = x**5  - y**5 + y**3

        # Assign the label based on the decision boundary
        if decision_boundary >= 0:
            p =1
        else:
            p = 0

        sample = np.array([x/2, x/2, y/3, y/3, y/3])
        sample = sample.reshape(1, self.d)
        val = [ p, 1-p ]

        # sample = np.array(sample)
        # mean = np.array([0.00052123, 0.00041331])
        # std = np.array([0.57674398, 0.57732179])
        # sample = ( sample - mean ) / std

        return sample , val 



##############################################################################
##############################################################################

def truncated_gaussian(mean, variance, a, b, size):
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(variance)

    # Calculate the lower and upper bounds for truncation
    lower_bound = (a - mean) / std_dev
    upper_bound = (b - mean) / std_dev

    # Generate samples from the truncated normal distribution
    samples = truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)

    return np.array(samples)

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

class LinearContexts:
    def __init__(self,  w, task):
        self.d = len(w) # number of features
        self.w = w
        self.type = 'linear'
        self.task = task
    
    def get_context(self, ):

        if self.task == 'imbalanced':
            context = truncated_gaussian(0, 0.1, 0, 1, self.d) if np.random.uniform(0,1)<0.5 else truncated_gaussian(1, 0.1, 0, 1, self.d)
        elif self.task == 'balanced':
            context = truncated_gaussian(0.5, 1, 0, 1, self.d)

        # cont = context.reshape(self.d,1)
        p = self.w @ context
        val = [ p, 1-p ]

        context = np.array(context)
        mean = np.array([0.49894511, 0.49964278, 0.49914392, 0.50041341, 0.49964411])
        std = np.array([0.31208973, 0.31217681, 0.31190383, 0.31202762, 0.31117992])
        context = ( context - mean ) / std

        return context , val 

class QuadraticContexts:
    def __init__(self,  w, task):
        self.d = len(w) # number of features
        self.w = w
        self.type = 'quadratic'
        self.task = task
    
    def get_context(self, ):

        if self.task == 'imbalanced':
            context = truncated_gaussian(0, 0.2, 0, 1, self.d ) if np.random.uniform(0,1)<0.5 else truncated_gaussian(1, 0.025, 0,  1, self.d )
        else:
            context = truncated_gaussian(1/np.sqrt(2), 0.05, 0,  1, self.d )

        cont = np.array(context)
        val =  (self.w @ cont)**2 
        val = [ val, 1-val ]

        mean = np.array([0.60673928, 0.60526987, 0.60531421, 0.60608301, 0.6049472 ])
        std = np.array([0.32465229, 0.32458575, 0.324807,   0.32434703, 0.32378323])
        cont = ( cont - mean ) / std

        return cont, val 

class NonLinearContexts:
    def __init__(self,  w, task):
        self.d = len(w) # number of features
        self.w = w
        self.type = 'nonlinear'
        self.task = task
    
    def get_context(self, ):

        if self.task == 'imbalanced':
            context = np.random.uniform(-10, 10, self.d) 
        # else:
            # context = truncated_gaussian(0, 5, -10,  10, self.d )

        cont = np.array(context)
        # print(cont)

        val =  (np.sin(  self.w @ context  ) + 1 ) / 2 
        val = [ val, 1-val ]

        mean = np.array([ 0.00210544, -0.02539498,  0.01848252, -0.00751817,  0.01412415])
        std = np.array([5.77238894, 5.77531259, 5.7664777,  5.77564962, 5.77461231])
        cont = ( cont - mean ) / std

        return cont, val 


class SinusoidContexts:
    def __init__(self,  w, task):
        self.d = len(w) # number of features
        self.w = w
        self.type = 'sinusoid'
        self.task = task
    
    def get_context(self, ):

        if self.task == 'imbalanced':
            if np.random.uniform(0,1)<0.725:
                context = truncated_gaussian(np.pi/self.d**2, 0.05, 0, np.pi, self.d) if np.random.uniform(0,1)<0.5 else  truncated_gaussian(0, 0.05, 0, np.pi, self.d)
            else :
                context = truncated_gaussian(np.pi/2, self.d * np.pi, 0, np.pi, self.d)
                
        else:
            context = truncated_gaussian(np.pi/6, 0.1, 0, np.pi, self.d) if np.random.uniform(0,1)<0.5 else  truncated_gaussian(5 * np.pi / 6, 0.1, 0, np.pi, self.d)
        
        
        cont = np.array(context) 
        val = np.sin(self.w@cont)
        val = [ val, 1-val ]

        mean = np.array([0.5812034 , 0.5774185 , 0.57832179, 0.57731911, 0.57838398])
        std = np.array([0.78190242, 0.77617083, 0.77710955, 0.77693384, 0.77836506])
        cont = ( cont - mean ) / std


        return cont, val 

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



# class MNISTcontexts():

#     def __init__(self, replacements, sampling_indexes):

#         self.horizon = len( sampling_indexes )
#         self.replacements = replacements 
#         self.sampling_indexes = sampling_indexes

#         self.switches = { '0':[6, 8, 9], '1':[4, 7], '2':[3], '3':[8, 9], '4':[8], '5':[6, 8], '6':[8], '7':[4, 8],'8':[0], '9':[8] }
        
#         # if digit_distribution == 'uniform':
#         self.digit_distribution =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#         # elif digit_distribution == 'gaussian':
#         #     self.digit_distribution = [0.00938744, 0.02826442, 0.06660327, 0.12284535, 0.17736299, 0.2004605, 0.17736299, 0.12284535, 0.06660327, 0.02826442]
#         # else:
#         #     self.digit_distribution = [3.03931336e-02, 1.21241604e-01, 1.92058772e-01, 1.21241604e-01, 3.03931336e-02, 2.99782587e-03, 1.15116446e-04, 5.01558658e-01, 1.53276385e-07, 2.01265511e-11]
        
#     def get_contexts(self, data, outcomes):
#         contexts = np.empty( ( self.horizon, 784) )
#         # stream = np.empty( ( horizon, 785) )
#         labels = np.zeros( self.horizon)
#         outcomes = np.zeros( self.horizon)

#         for i, index, in enumerate( range( len(self.sampling_indexes) ) ) :

#             outcome = outcomes[i]
#             X, y =  data[index]
#             X = X.numpy()

#             if outcome == 1:
#                 #attacked_digit =  np.random.choice( [0,1,2,3,4,5,6,7,8,9], p= digit_distribution )
#                 candidates = self.switches[str(y)]
#                 replacement_digit = np.random.choice( candidates , p= np.ones( len(candidates)  ) / len(candidates) )
#                 choice = np.random.randint( 0, len(self.replacements[ replacement_digit ])-1 )
#                 replaced_image = self.replacements[replacement_digit][choice]
#                 X = replaced_image
#             contexts[i] = X.flatten()
#             labels[i] = y

#         return contexts
