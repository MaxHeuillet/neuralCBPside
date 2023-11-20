import numpy as np
from .strategy import Strategy
import torch
import copy
from sklearn.random_projection import GaussianRandomProjection
import os
import time



class Vessal():

    def __init__(self, game, nclasses, m, device):

        self.name = 'vessal'
        self.device = device

        self.game = game
        self.nclasses = nclasses
        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = None#geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)

        self.m = m
        self.H = 50

    def reset(self, d):
        self.d = d
        if self.nclasses == 2:
            self.func = DeployedNetwork( self.d , self.m, 1).to(self.device)
        else:
            input_dim = self.d + (self.nclasses-1) * self.d
            self.func = DeployedNetwork( input_dim , self.m, 1).to(self.device)

        self.func0 = copy.deepcopy(self.func)
        self.hist = CustomDataset()
    
    def encode_context(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        ci = torch.zeros(1, self.d).to(self.device)
        x_list = []
        for k in range(self.nclasses):
            inputs = []
            for l in range(k):
                inputs.append(ci)
            inputs.append(X)
            for l in range(k+1, self.nclasses):
                inputs.append(ci)
            inputs = torch.cat(inputs, dim=1).to(torch.float32)
            x_list.append(inputs)
        return x_list

    def get_action(self, t, X):

        if self.nclasses == 2:
            prediction = self.func( torch.from_numpy( X ).float().to(self.device) ).cpu().detach()
            probability = expit(prediction)
            self.pred_action = 1 if probability < 0.5 else 2
        else:
            self.u_list = []
            self.x_list = self.encode_context(X)
            for k in range(self.nclasses):
                prediction = self.func( self.x_list[k] ).cpu().detach()
                #print(k, prediction)
                self.u_list.append((k, prediction.item() ))
                
            self.u_list = sorted(self.u_list, key=lambda x: x[1], reverse=True)
            #print(self.u_list)
            self.pred_action = self.u_list[0][0] + 1

        
        action = 0 if random.random() < 0.1 else self.pred_action

        explored = 1 if action ==0 else 0

        history = {'monitor_action':action, 'explore':explored,}
            
        return action, history

    def update(self, action, feedback, outcome, t, X):

        if action == 0:
            if self.nclasses == 2:
                self.hist.append( X , [outcome] )
            else:
                for k in range(self.nclasses):
                    if k == outcome:
                        self.hist.append( self.x_list[k].detach().cpu(), [1] )
                    else: 
                        self.hist.append( self.x_list[k].detach().cpu(), [0] )

            
        global_loss = []
        global_losses = []
        if (t>self.N):
            if (t % 50 == 0 and t<1000) or (t % 500 == 0 and t>=1000):

                self.func = copy.deepcopy(self.func0)
                optimizer = optim.Adam(self.func.parameters(), lr=0.001, weight_decay = 0 )
                dataloader = DataLoader(self.hist, batch_size=1000, shuffle=True) 
                #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
                if self.nclasses == 2:
                    loss = nn.BCEWithLogitsLoss()
                else:
                    loss = nn.MSELoss()

                for _ in range(1000): 
                        
                    train_loss, losses = self.step(dataloader, loss, optimizer)
                    current_lr = optimizer.param_groups[0]['lr']
                    global_loss.append(train_loss)
                    global_losses.append(losses)
                    # if _ % 10 == 0 :
                    #     scheduler.step()
                    # scheduler.step()
                    if _ % 25 == 0:
                        print('train loss', train_loss, 'losses', losses )

        return global_loss, global_losses
                

    def step(self, loader, loss_func, opt):
        #""Standard training/evaluation epoch over the dataset"""

        for X, y in loader:
            X = X.to(self.device).float()
            #if self.nclasses == 2:
            y = y.to(self.device).float()
            #else:
            #    y = one_hot(y, num_classes=10).to(self.device).float()
            
            #print(y)
            loss = 0
            losses = []
            losses_vec =[]
 

            pred = self.func(X).squeeze(1)
            #print('pred shape',pred.shape, y.shape)


            l = loss_func(pred, y)
            loss += l
            losses.append( l )
            losses_vec.append(l.item())

            opt.zero_grad()
            l.backward()
            opt.step()
            # print(losses)
        return loss.item(), losses_vec



















class StreamingSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(StreamingSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.skipped = []

        if self.args["data"] == 'CLOW' or self.args["data"] == 'clip':
            self.transformer = GaussianRandomProjection(n_components=2560)
        self.zeta = self.args["zeta"]

    # just in case values get too big, sometimes happens
    def inf_replace(self, mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat

    def streaming_sampler(self, samps, k, early_stop=False, streaming_method='det',  cov_inv_scaling=100, embs="grad_embs"):
        inds = []
        skipped_inds = []
        if embs == "penultimate":
            samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        dim = samps.shape[-1]
        rank = samps.shape[-2]

        covariance = torch.zeros(dim,dim).cuda()
        covariance_inv = cov_inv_scaling * torch.eye(dim).cuda()
        samps = torch.tensor(samps)
        samps = samps.cuda()

        for i, u in enumerate(samps):
            if i % 1000 == 0: print(i, len(inds), flush=True)
            if rank > 1: u = torch.Tensor(u).t().cuda()
            else: u = u.view(-1, 1)
            
            # get determinantal contribution (matrix determinant lemma)
            if rank > 1:
                norm = torch.abs(torch.det(u.t() @ covariance_inv @ u))
            else:
                norm = torch.abs(u.t() @ covariance_inv @ u)

            ideal_rate = (k - len(inds))/(len(samps) - (i))
            # just average everything together: \Sigma_t = (t-1)/t * A\{t-1}  + 1/t * x_t x_t^T
            covariance = (i/(i+1))*covariance + (1/(i+1))*(u @ u.t())

            self.zeta = (ideal_rate/(torch.trace(covariance @ covariance_inv))).item()

            pu = np.abs(self.zeta) * norm

            if np.random.rand() < pu.item():
                inds.append(i)
                if early_stop and len(inds) >= k:
                    break
                
                # woodbury update to covariance_inv
                inner_inv = torch.inverse(torch.eye(rank).cuda() + u.t() @ covariance_inv @ u)
                inner_inv = self.inf_replace(inner_inv)
                covariance_inv = covariance_inv - covariance_inv @ u @ inner_inv @ u.t() @ covariance_inv
            else:
                skipped_inds.append(i)

        return inds, skipped_inds


    def get_valid_candidates(self):
        skipped = np.zeros(self.n_pool, dtype=bool)
        skipped[self.skipped] = True
        if self.args["single_pass"]:
            valid = ~self.idxs_lb & ~skipped & self.allowed 
        else:
            valid = ~self.idxs_lb 
        return valid 


    def query(self, n):#, num_round=0):

        valid = self.get_valid_candidates()
        idxs_unlabeled = np.arange(self.n_pool)[valid]

        rank = self.args["rank"]
        if self.args["embs"] == "penultimate":
            gradEmbedding = self.get_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
            # print('pen embedding shape: {}'.format(gradEmbedding.shape))
        else:
            gradEmbedding = self.get_exp_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], rank=rank).numpy()
            # print('gradient embedding shape: {}'.format(gradEmbedding.shape))

        early_stop = self.args["early_stop"] 
        cov_inv_scaling = self.args["cov_inv_scaling"]
       
        start_time = time.time()
        chosen, skipped = self.streaming_sampler(gradEmbedding, n, early_stop=early_stop, \
            cov_inv_scaling=cov_inv_scaling, embs = self.args["embs"])
        print(len(idxs_unlabeled), len(chosen), flush=True)
        print('compute time (sec):', time.time() - start_time, flush=True)
        print('chosen: {}, skipped: {}, n:{}'.format(len(chosen),len(skipped),n), flush=True)

        # If more than n samples were selected, take the first n.
        if len(chosen) > n:
            chosen = chosen[:n]

        self.skipped.extend(idxs_unlabeled[skipped])

        result = idxs_unlabeled[chosen]
        if self.args["fill_random"]:
            # If less than n samples where selected, fill is with random samples.
            if len(chosen) < n:
                labelled = np.copy(self.idxs_lb)
                labelled[idxs_unlabeled[chosen]] = True
                remaining_unlabelled = np.arange(self.n_pool)[~labelled]
                n_random = n - len(chosen)
                fillers = remaining_unlabelled[np.random.permutation(len(remaining_unlabelled))][:n_random]
                result = np.concatenate([idxs_unlabeled[chosen], fillers], axis=0)

        return result