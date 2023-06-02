import numpy as np
import geometry_v3

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

        self.name = 'neuralcbpside'

        self.game = game
        self.d = d
        self.N = game.n_actions
        self.M = game.n_outcomes

        self.SignalMatrices = game.SignalMatrices
        self.pareto_actions = geometry_v3.getParetoOptimalActions(game.LossMatrix, self.N, self.M, [])
        self.mathcal_N = game.mathcal_N
        self.N_plus =  game.N_plus
        self.V = game.V
        self.v = game.v 
        self.W = self.getConfidenceWidth( )
        self.alpha = alpha
        self.lbd = lbd
        self.eta =  self.W **2/3 
        self.A = geometry_v3.alphabet_size(game.FeedbackMatrix_PMDMED, game.N, game.M)

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
        # self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        # self.U = self.lbd * torch.ones((self.total_param,)).cuda()
        # self.zeta = 1
        self.Zt = np.identity() 

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


            # for i in range(self.N):
            #     pred =  self.contexts[i]['weights']( torch.from_numpy(X.T).float().cuda() ) 
            #     q.append( pred.cpu().detach().numpy().T )
            #     factor = self.d * (  np.sqrt( (self.d+1) * np.log(t)  ) + len(self.SignalMatrices[i]) )
            #     width = X.T @ self.contexts[i]['V_it_inv'] @ X 
            #     formule = factor * width
            #     w.append( formule )

            # print(w)
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
            result =  geometry_v3.getParetoOptimalActions(self.game.LossMatrix, self.N, self.M, halfspace)
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
            result =  geometry_v3.getNeighborhoodActions(self.game.LossMatrix, self.N, self.M, halfspace,  self.mathcal_N )
            self.memory_neighbors[code ] =result
 
        return result

