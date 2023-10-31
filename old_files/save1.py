import numpy as np
import geometry_gurobi

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

class Network(nn.Module):
    def __init__(self, output_dim, dim, hidden_size=10):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        return self.fc2(self.activate(self.fc1(x)))


def convert_list(A):
    B = []
    B.append(np.array([A[0]]).reshape(1, 1))
    sub_array = np.array(A[1:]).reshape(2, 1)
    B.append(sub_array)
    return B

class NeuralCBPside():

    def __init__(self, game, d, alpha, lbd, hidden):

        self.name = 'NeuralCBPsidev1'

        self.game = game
        self.d = d
        self.N = game.n_actions
        self.M = game.n_outcomes

        self.SignalMatrices = game.SignalMatrices
        self.pareto_actions = geometry_gurobi.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        self.N_plus =  game.N_plus
        self.V = game.V
        self.v = game.v 
        self.W = self.getConfidenceWidth( )
        self.alpha = alpha
        self.lbd = lbd
        self.eta =  self.W **2/3 
        self.A = geometry_gurobi.alphabet_size(game.FeedbackMatrix_PMDMED, game.N, game.M)

        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)] 
        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.hidden = hidden

        self.contexts = []
        for i in range(self.N):
            output_dim = len( set(self.game.FeedbackMatrix[i]) )
            func = Network( output_dim, self.d, hidden_size=self.hidden).cuda()
            self.contexts.append( {'features':[], 'labels':[], 'weights': func, 'V_it_inv': np.identity(self.d) } )

        self.lbd = lbd

    def set_nlabels(self, nlabels):
        self.d = nlabels

    def getConfidenceWidth(self, ):
        W = np.zeros(self.N)
        for pair in self.mathcal_N:
            for k in self.V[ pair[0] ][ pair[1] ]:
                vec = self.v[ pair[0] ][ pair[1] ][k]
                W[k] = np.max( [ W[k], np.linalg.norm(vec ) ] )
        return W

    def reset(self,):
        self.n = np.zeros( self.N )
        self.nu = [ np.zeros(   ( len( set(self.game.FeedbackMatrix[i]) ),1)  ) for i in range(self.N)]  #[ np.zeros(    len( np.unique(self.game.FeedbackMatrix[i] ) )  ) for i in range(self.N)] 
        self.memory_pareto = {}
        self.memory_neighbors = {}
        self.contexts = []
        for i in range(self.N):
            output_dim = len( set(self.game.FeedbackMatrix[i]) )
            func = Network( output_dim, self.d, hidden_size=self.hidden).cuda()
            self.contexts.append( {'features':[], 'labels':[], 'weights': func, 'V_it_inv': np.identity(self.d) } )

    def update_A_inv(self):
        self.A_inv[self.action] = inv_sherman_morrison( self.grad_approx[self.action], self.A_inv[self.action] )

    def get_action(self, t, X):

        if t < self.N: # jouer chaque action une fois au debut du jeu
            action = t

        else: 
            
            g_list = []
            halfspace = []
            q = []
            w = []
                        
            for i in range(self.N):
                
                self.contexts[i]['weights'].zero_grad()
                pred =  self.contexts[i]['weights']( torch.from_numpy(X.T).float().cuda() ) 
                pred.backward()

                g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])
                g_list.append(g)
                sigma2 = self.lamdba * self.nu * g * g / self.U
                sigma = torch.sqrt(torch.sum(sigma2))


                
                factor = self.confidence_multiplier
                width = np.sqrt(np.dot(self.grad_approx[i], np.dot(self.A_inv[i], self.grad_approx[i].T)) )
                formule = factor * width
                w.append( formule )

                
                q.append( pred.cpu().detach().numpy().T )

            for pair in self.mathcal_N:
                tdelta = np.zeros( (1,) )
                c = 0

                for k in  self.V[ pair[0] ][ pair[1] ]:
                    tdelta += self.v[ pair[0] ][ pair[1] ][k].T @ q[k]
                    c += np.linalg.norm( self.v[ pair[0] ][ pair[1] ][k] ) * w[k] 

                tdelta = tdelta[0]
                print('pair', pair, 'tdelta', tdelta, 'confidence', c)
                if( abs(tdelta) >= c):
                    halfspace.append( ( pair, np.sign(tdelta) ) ) 

            P_t = self.pareto_halfspace_memory(halfspace)
            N_t = self.neighborhood_halfspace_memory(halfspace)

            Nplus_t = []
            for pair in N_t:
                Nplus_t.extend( self.N_plus[ pair[0] ][ pair[1] ] )
            Nplus_t = np.unique(Nplus_t)

            V_t = []
            for pair in N_t:
                V_t.extend( self.V[ pair[0] ][ pair[1] ] )
            V_t = np.unique(V_t)

            R_t = []
            
            for k in V_t:
              val =  X.T @ self.contexts[k]['V_it_inv'] @ X
              t_prime = t
              with np.errstate(divide='ignore'): 
                rate = np.sqrt( self.eta[k] * self.N**2 * 4 *  self.d**2  *(t_prime**(2/3) ) * ( self.alpha * np.log(t_prime) )**(1/3) ) 
                if val[0][0] > 1/rate : 
                    R_t.append(k)
    
            union1= np.union1d(  P_t, Nplus_t )
            union1 = np.array(union1, dtype=int)

            S =  np.union1d(  union1  , R_t )
            S = np.array( S, dtype = int)
            S = np.unique(S)

            values = { i:self.W[i]*w[i] for i in S}
            action = max(values, key=values.get)

        return action

    def update(self, action, feedback, outcome, t, X):

        

        e_y = np.zeros( (self.M, 1) )
        e_y[outcome] = 1
        Y_t =  self.game.SignalMatrices[action] @ e_y 

        self.contexts[action]['labels'].append( torch.from_numpy(Y_t.T).float() )
        self.contexts[action]['features'].append( torch.from_numpy(X.T).float() )
       
        # Y_it =  np.squeeze(Y_it, 2).T 
        # X_it =  np.squeeze(X_it, 2).T 
        # Y_it = torch.from_numpy(Y_t).long().cuda()
        # X_it = torch.from_numpy(X_it).float().cuda()
        # print(X_it)

        V_it_inv = self.contexts[action]['V_it_inv']
        self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 

                
        optimizer = optim.SGD(self.contexts[action]['weights'].parameters(), lr=1e-2, weight_decay=self.lbd)
        length = len(self.contexts[action]['labels'])
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        loss_func = nn.MSELoss()
        while True:
            batch_loss = 0
            for idx in index:
                c = self.contexts[action]['features'][idx]
                f = self.contexts[action]['labels'][idx].cuda()
                pred = self.contexts[action]['weights']( c.cuda() )
                # print(c, f, pred)
                optimizer.zero_grad()
                # print(c.shape)
                loss = loss_func(pred, f)
                # print(loss)
                # delta = pred - f #difference entre le gain predit et le gain qui a ete recu
                # loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 100:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length



        

    def halfspace_code(self, halfspace):
        string = ''
        for element in halfspace:
            pair, sign = element
            string += '{}{}{}'.format(pair[0],pair[1], sign)
        return string 


    def pareto_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_pareto.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_pareto[ code ]
        else:
            result =  geometry_gurobi.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
            self.memory_pareto[code ] =result
 
        return result

    def neighborhood_halfspace_memory(self,halfspace):

        code = self.halfspace_code(  sorted( halfspace) )
        known = False
        for mem in self.memory_neighbors.keys():
            if code  == mem:
                known = True

        if known:
            result = self.memory_neighbors[ code ]
        else:
            result =  geometry_gurobi.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )
            self.memory_neighbors[code ] =result
 
        return result





















# Define the range of x and y values for the grid
x_min, x_max = -1, 1
y_min, y_max = -1, 1

# Generate a grid of points
num_points = 1000
x_values = np.linspace(x_min, x_max, num_points)  
y_values = np.linspace(y_min, y_max, num_points)  
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Compute the decision boundary for the grid of points
b = 0.15

plt.figure(figsize=(4, 4))

decision_boundary_grid = context_generator.decision_boundary_function(x_grid, y_grid, b)
plt.contourf(x_grid, y_grid, decision_boundary_grid, levels=1, alpha=0.6, cmap=plt.cm.coolwarm)

contexts = np.array( [ context_generator.denormalize(i[4]) for i in train_hist ] ).squeeze(1) 

action0 = [ i[0] if i[0]==2 else np.nan for i in train_hist ]
indices_action0 = np.where(~np.isnan(action0))[0]
contexts0 = contexts[indices_action0]

action1 = [ i[0] if i[0]==1 else np.nan for i in train_hist ]
indices_action1 = np.where(~np.isnan(action1))[0]
contexts1 = contexts[indices_action1]

action2 = [ i[0] if i[0]==0 else np.nan for i in train_hist ]
indices_action2 = np.where(~np.isnan(action2))[0]
contexts2 = contexts[indices_action2]

# plt.plot(contexts0[:,0], contexts0[:,1], '.', color = 'orange', markersize = 2, label = 'predicted as class 1')
# plt.plot(contexts1[:,0], contexts1[:,1], '.', color = 'blue', markersize = 2, label = 'predicted as class 2')
plt.plot(contexts2[:,0], contexts2[:,1], '.', color = 'green', markersize = 2, label = 'explored')

plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.ylim((-1,1))
plt.xlim((-1,1))
# plt.title('Training decision boundary (shift = {})'.format(b))
plt.title('Deployment decision boundary (shift = {})'.format(b))
plt.legend(loc = (-0.4,-0.25),ncol = 3)
# Save the figure to a file with tight layout and 380 DPI
# plt.savefig('./figures/CBP_DB_{}.png'.format(b), dpi=380, bbox_inches='tight')
plt.savefig('./figures/ETC_exploration3_{}.png'.format(b), dpi=380, bbox_inches='tight')
# plt.savefig('./figures/CBP_exploration3_{}.png'.format(b), dpi=380, bbox_inches='tight')












# Define the range of x and y values for the grid
x_min, x_max = -1, 1
y_min, y_max = -1, 1

# Generate a grid of points
num_points = 1000
x_values = np.linspace(x_min, x_max, num_points)  
y_values = np.linspace(y_min, y_max, num_points)  
x_grid, y_grid = np.meshgrid(x_values, y_values)

# Compute the decision boundary for the grid of points
for b in [0.15]: #[0, 0.15]

    plt.figure(figsize=(4, 4))

    decision_boundary_grid = context_generator.decision_boundary_function(x_grid, y_grid, b)

    plt.contourf(x_grid, y_grid, decision_boundary_grid, levels=1, alpha=0.6, cmap=plt.cm.coolwarm)

    contexts = np.array( [ context_generator.denormalize(i[4]) for i in depl_hist ] ).squeeze(1) 

    action0 = [ i[0] if i[0]==2 else np.nan for i in depl_hist ]
    indices_action0 = np.where(~np.isnan(action0))[0]
    contexts0 = contexts[indices_action0]
    action1 = [ i[0] if i[0]==1 else np.nan for i in depl_hist ]
    indices_action1 = np.where(~np.isnan(action1))[0]
    contexts1 = contexts[indices_action1]

    plt.plot(contexts0[:,0], contexts0[:,1], '.', markersize = 1, color = 'red')
    plt.plot(contexts1[:,0], contexts1[:,1], '.', markersize = 1, color = 'blue')

    # plt.scatter(contexts[indices_predaction0][:,0], contexts[indices_predaction0][:,1], s = 1, color='blue', label='Predicted Points')
    # plt.scatter(contexts[indices_predaction1][:,0], contexts[indices_predaction1][:,1], s = 1, color='red', label='Predicted Points')

    # Add labels and title to the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title(' Decision Boundary')
    # Adjust the layout for better spacing
    plt.tight_layout()
    plt.ylim((-1,1))
    plt.xlim((-1,1))
    plt.title('Deployment decision boundary (shift = {})'.format(b))

    # Save the figure to a file with tight layout and 380 DPI
    plt.savefig('./figures/ETC_exploitation3_{}.png'.format(b), dpi=380, bbox_inches='tight')
    # plt.savefig('./figures/CBP_exploitation3_{}.png'.format(b), dpi=380, bbox_inches='tight')


from matplotlib.ticker import ScalarFormatter

new_global_loss = np.vstack( [ i for i in global_losses if len(i)>0 ] )

plt.figure(figsize=(8, 4))

plt.yscale('log')
plt.grid(color='gray', linestyle='-')
plt.xlim( (-10, 19000) )

plt.xticks(custom_ticks, custom_tick_labels, rotation=45)

# Set tick locations and labels for the y-axis
tick_locations = [0.01, 0.1, 1, 10,]  # Define your desired tick locations
tick_labels = ['0.01', '0.1', '1', '10', ]  # Corresponding labels
ax = plt.gca()
ax.yaxis.set_major_locator(plt.FixedLocator(tick_locations))
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_yticklabels(tick_labels)

plt.plot( new_global_loss[:,0], label = 'symbole 0' )
plt.plot( new_global_loss[:,1], label = 'symbole 1' )
plt.plot( new_global_loss[:,2], label = 'symbole 2' )
plt.plot( new_global_loss[:,3], label = 'symbole 3' )

plt.xlabel('Step + 1000 epochs')
plt.ylabel('Loss')
plt.legend()

# plt.savefig('./figures/loss_evolution_{}.png'.format(idx), dpi=380, bbox_inches='tight')
