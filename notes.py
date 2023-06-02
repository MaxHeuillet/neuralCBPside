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