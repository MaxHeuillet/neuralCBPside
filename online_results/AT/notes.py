                    # unit_vectors = torch.eye(output_dim).cuda() #

                    # # print( 'context',torch.from_numpy(X).squeeze(1).float().cuda().shape )
                    # # print( 'unit', unit_vectors[0].shape )
                    # # print( 'pred', self.functionnal[i]['weights'].forward(torch.from_numpy(X).squeeze(1).float().cuda() ) )
                    # _, vjp_fn = torch.func.vjp(partial(self.functionnal[i]['weights'].forward),  torch.from_numpy(X).squeeze(1).float().cuda()  )

                    # ft_jacobian = torch.vmap(vjp_fn)(unit_vectors)[0]

                    # # print('jac shape',ft_jacobian)
                    # ft_jacobian = ft_jacobian.detach().cpu().numpy()

                    # # print(ft_jacobian.shape)
                    # avg_gradients = np.mean(ft_jacobian, 0)

                    # print(avg_gradients.shape)
                    # self.g_list.append(avg_gradients)
                
                # print(g.shape)
                # print(self.functionnal[i]['Z_it_inv'].shape)

# def get_combinations(A):
#     identity_matrix = torch.eye(A)
#     combinations = list(itertools.combinations(identity_matrix, A))[0]
#     return torch.stack(combinations).to(self.device)

# def convert_list(A):
#     B = []
#     B.append(np.array([A[0]]).reshape(1, 1))
#     sub_array = np.array(A[1:]).reshape(2, 1)
#     B.append(sub_array)
#     return B


# with gzip.open( './results/{}/benchmark_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach) ,'ab') as g:

#     for jobid in range(n_folds):

#         pkl.dump( result[jobid], g)

#     with gzip.open(  './results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach, jobid) ,'rb') as f:
#         r = pkl.load(f)

# bashCommand = 'rm ./results/{}/benchmark_{}_{}_{}_{}_{}_{}.pkl.gz'.format(args.game, args.task, args.context_type, horizon, n_folds, args.approach, jobid)
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()

# algos_dico = {
#           'neuralcbp_theory':neuralcbpside_v3.NeuralCBPside(game, 'theory', 1.01, 0.05),
#           'neuralcbp_simplified':neuralcbpside_v3.NeuralCBPside(game, 'simplified', 1.01, 0.05),
#           'neuralcbp_1':neuralcbpside_v3.NeuralCBPside(game, '1', 1.01, 0.05)  }
#'CBPside':cbpside.CBPside(game, dim, factor_choice, 1.01, 0.05),