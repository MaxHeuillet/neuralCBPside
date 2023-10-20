import numpy as np
import collections
import pulp as lp
from multiprocessing import Pool

import gurobipy as gp
from gurobipy import GRB



def alphabet_size(FeedbackMatrix, N,M):
    alphabet = []
    for i in range(N):
        for j in range(M):
            alphabet.append(FeedbackMatrix[i][j])
    return len(set(alphabet)) 


##########################################################################################
############################################# PULP:
##########################################################################################

# def solve_LP(args):
#     z, M, LossMatrix, halfspace = args
#     vars = [lp.LpVariable('p_{}'.format(i), 0.00001, 1.0, lp.LpContinuous) for i in range(M)]
#     m = lp.LpProblem(name="Pareto_Optimization_{}".format(z), sense=lp.LpMinimize)
#     m += (lp.lpSum(vars) == 1.0, "css")
    
#     lossExprs = np.dot(LossMatrix - LossMatrix[z, :], vars)
#     for i2, lossExpr in enumerate(lossExprs):
#         if i2 != z:
#             m += (lossExpr >= 0.0, 'c_{}'.format(i2))
    
#     for element in halfspace:
#         pair, sign = element
#         if sign != 0:
#             halfspaceExpr = lp.lpSum(sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M))
#             m += (halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))
    
#     m.solve()
#     if lp.LpStatus[m.status] == "Optimal":
#         return z
#     return None



# def getParetoOptimalActions(LossMatrix, N, M, halfspace, num_pools=None):
#     lp.LpSolverDefault.msg = 0
#     LossMatrix = np.array(LossMatrix)
    
#     with Pool(processes=num_pools) as pool:
#         results = pool.map(solve_LP, [(z, M, LossMatrix, halfspace) for z in range(N)])

#     return [z for z in results if z is not None]


# def parallel_check(args):
#     LossMatrix, N, M, halfspace, pair = args
#     i1, i2 = pair
#     if isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
#         return [i1, i2]
#     return None

# def getNeighborhoodActions(LossMatrix, N, M, halfspace, mathcal_N, num_pools=None):
#     with Pool(processes=num_pools) as pool:
#         results = pool.map(parallel_check, [(LossMatrix, N, M, halfspace, pair) for pair in mathcal_N])
#     actions = [result for result in results if result is not None]
#     return actions

# def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
#     lp.LpSolverDefault.msg = 0  # or False

#     m = lp.LpProblem(name="Neighbor_Check", sense=lp.LpMinimize)

#     # Define variables
#     vars = [lp.LpVariable("p_{}".format(j), 0.00001, 1.0, lp.LpContinuous) for j in range(M)]

#     # Add simplex constraint
#     m += (lp.lpSum(vars) == 1.0, "css")

#     # Add two-degenerate constraint
#     twoDegenerateExpr = lp.lpSum((LossMatrix[i2][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
#     m += (twoDegenerateExpr == 0.0, "cdeg")

#     # Add loss constraints
#     for i3 in range(N):
#         if i3 != i1:
#             lossExpr = lp.lpSum((LossMatrix[i3][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
#             m += (lossExpr >= 0.0, "c_{}".format(i3))

#     # Add halfspace constraints
#     for element in halfspace:
#         pair, sign = element[0], element[1]
#         if sign != 0:
#             halfspaceExpr = lp.lpSum(
#                 sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M)
#             )
#             m += (halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))

#     m.solve()
#     return lp.LpStatus[m.status] == "Optimal"


##########################################################################################
############################################# GUROBI:
##########################################################################################



def solve_LP(args):
    z, M, LossMatrix, halfspace = args
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            m.setParam('Threads', 1)
            # print(f'Number of cores that will be used: {m.Params.Threads}')
            # print(f'Number of available cores: {m.Params.ConcurrentMIP}')
            # Create a new model
            #m = gp.Model() #name="Pareto_Optimization_{}".format(z)

            m.Params.LogToConsole = 0
            
            m.setParam("OutputFlag", 0)  # turn off Gurobi's output
            #print('n threads', m.Params.Threads)
            #m.setParam("Threads", 1)
            #print('n threads', m.Params.Threads)

            # Add variables
            vars = m.addVars(M, lb=0.00001, ub=1.0, vtype=gp.GRB.CONTINUOUS, name='p')

            # Set objective (in this case, no objective is specified, so we'll set it to 0)
            m.setObjective(0, gp.GRB.MINIMIZE)

            # Add constraint: sum(vars) == 1.0
            m.addConstr(sum(vars[i] for i in range(M)) == 1.0, "css")

            # Add loss constraints
            lossExprs = np.dot(LossMatrix - LossMatrix[z, :], [vars[i] for i in range(M)])
            for i2, lossExpr in enumerate(lossExprs):
                if i2 != z:
                    m.addConstr(lossExpr >= 0.0, 'c_{}'.format(i2))

            # Add halfspace constraints
            for element in halfspace:
                pair, sign = element
                if sign != 0:
                    halfspaceExpr = sum(sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M))
                    m.addConstr(halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))

            # Solve the model
            m.optimize()

            if m.status == gp.GRB.OPTIMAL:
                return z
    return None

def getParetoOptimalActions(LossMatrix, N, M, halfspace, num_pools=None):
    LossMatrix = np.array(LossMatrix)

    with Pool(processes=num_pools) as pool:
        results = pool.map(solve_LP, [(z, M, LossMatrix, halfspace) for z in range(N)])

    return [z for z in results if z is not None]




def parallel_check(args):
    LossMatrix, N, M, halfspace, pair = args
    i1, i2 = pair
    if isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
        return [i1, i2]
    return None

def getNeighborhoodActions(LossMatrix, N, M, halfspace, mathcal_N, num_pools=None):
    with Pool(processes=num_pools) as pool:
        results = pool.map(parallel_check, [(LossMatrix, N, M, halfspace, pair) for pair in mathcal_N])
    actions = [result for result in results if result is not None]
    return actions

def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:

            m.Params.LogToConsole = 0
            lp.LpSolverDefault.msg = 0
            m.setParam("OutputFlag", 0)  
            # Turn off Gurobi's output
            #print('n threads', m.Params.Threads)
            #m.setParam("Threads", 1)
            #print('n threads', m.Params.Threads)

            # Define variables
            vars = m.addVars(M, lb=0.00001, ub=1.0, vtype=gp.GRB.CONTINUOUS, name='p')

            # Add simplex constraint
            m.addConstr(sum(vars[j] for j in range(M)) == 1.0, "css")

            # Add two-degenerate constraint
            twoDegenerateExpr = sum((LossMatrix[i2][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
            m.addConstr(twoDegenerateExpr == 0.0, "cdeg")

            # Add loss constraints
            for i3 in range(N):
                if i3 != i1:
                    lossExpr = sum((LossMatrix[i3][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
                    m.addConstr(lossExpr >= 0.0, "c_{}".format(i3))

            # Add halfspace constraints
            for element in halfspace:
                pair, sign = element[0], element[1]
                if sign != 0:
                    halfspaceExpr = sum(
                        sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M)
                    )
                    m.addConstr(halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))

            m.optimize()
            if m.status == gp.GRB.OPTIMAL:
                result = True
            else: 
                result = False

    return result
































# def getParetoOptimalActions(LossMatrix, N, M, halfspace):
#     actions = []
#     for z in range(N):
#         feasible = True

#         m = gp.Model( )
#         m.Params.LogToConsole = 0

#         vars = []
#         for i in range(M):
#             varName =  'p_{}'.format(i) 
#             vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName) )
#             m.update()

#         simplexExpr = gp.LinExpr()
#         for j in range(M):
#             simplexExpr += 1.0 * vars[j]

#         m.addConstr(simplexExpr == 1.0, "css")

#         for i2 in range(N):
#             if(i2 != z):
#                 lossExpr = gp.LinExpr()
#                 for j in range(M):
#                     lossExpr += ( LossMatrix[i2][j] - LossMatrix[z][j] ) * vars[j]
#                     lossConstStr = 'c {}'.format(i2)
#                 m.addConstr(lossExpr >= 0.0, lossConstStr )

#         for element in halfspace:
#             pair, sign = element[0], element[1]
#             if sign != 0:
#                 halfspaceExpr = gp.LinExpr()
#                 for j in range(M):
#                     coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
#                     if coef != 0:
#                         halfspaceExpr += coef * vars[j]
#                 halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
#                 m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
#         try:
#             m.optimize()
#             objval = m.objVal
#         except:
#             feasible=False

#         if feasible:
#             actions.append(z)

#     return actions

# def getNeighborhoodActions(LossMatrix, N, M, halfspace,mathcal_N):
#     actions = []
#     for pair in mathcal_N:
#         i1,i2 = pair
#         if isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
#             actions.append( [i1,i2] )
#     return actions

# def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
#     feasible = True
#     lp.LpSolverDefault.msg = 0  # or False

#     m = lp.LpProblem(name="Neighbor_Check", sense=lp.LpMinimize)

#     vars = []
#     for j in range(M):
#         varName = "p_{}".format(j)
#         var = lp.LpVariable(varName, 0.00001, 1.0, lp.LpContinuous)
#         vars.append(var)

#     simplexExpr = lp.lpSum(vars)
#     m += (simplexExpr == 1.0, "css")

#     twoDegenerateExpr = lp.lpSum((LossMatrix[i2][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
#     m += (twoDegenerateExpr == 0.0, "cdeg")

#     for i3 in range(N):
#         if i3 == i1 or i2 == i1:
#             continue
#         lossExpr = lp.lpSum((LossMatrix[i3][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
#         lossConstStr = "c_{}".format(i3)
#         m += (lossExpr >= 0.0, lossConstStr)

#     for element in halfspace:
#         pair, sign = element[0], element[1]
#         if sign != 0:
#             halfspaceExpr = lp.lpSum(
#                 sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M)
#             )
#             halfspaceConstStr = "ch_{}_{}".format(pair[0], pair[1])
#             m += (halfspaceExpr >= 0.001, halfspaceConstStr)

#     m.solve()
#     if lp.LpStatus[m.status] != "Optimal":
#         feasible = False

#     return feasible

# def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
#     feasible = True


#     m = gp.Model( )
#     m.Params.LogToConsole = 0
#     vars = []
#     for j in range(M):
#         varName = "p {}".format(j)
#         vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName ) )
#         m.update()

#     simplexExpr = gp.LinExpr()
#     for j in range(M):
#         simplexExpr += 1.0 * vars[j]
#     m.addConstr(simplexExpr == 1.0, "css") 

#     twoDegenerateExpr = gp.LinExpr()
#     for j in range(M):
#         twoDegenerateExpr += (LossMatrix[i2][j]-LossMatrix[i1][j]) * vars[j]
#     m.addConstr(twoDegenerateExpr == 0.0, "cdeg")

#     for i3 in range(N):
#         if( (i3 == i1) or (i2 == i1) ):
#             pass
#         else:
#             lossExpr = gp.LinExpr()
#             for j in range(M):
#                 lossExpr += ( LossMatrix[i3][j]-LossMatrix[i1][j] ) * vars[j]
#             lossConstStr = "c".format(i3)
#             m.addConstr(lossExpr >= 0.0, lossConstStr )

#     for element in halfspace:
#         pair, sign = element[0], element[1]
#         if sign != 0:
#             halfspaceExpr = gp.LinExpr()
#             for j in range(M):
#                 coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
#                 if coef != 0:
#                     halfspaceExpr += coef * vars[j]
#             halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
#             m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
#     try:
#         m.optimize()
#         objval = m.objVal
#     except:
#         feasible = False

#     return feasible






















# def calculate_signal_matrices(FeedbackMatrix, N,M,A):
#     signal_matrices = []
#     for i in range(N):
#         signalMatrix = np.zeros( (A,M) )
#         for j in range(M):
#             a = FeedbackMatrix[i][j]
#             signalMatrix[a][j] = 1
#         signal_matrices.append(signalMatrix)
#     return signal_matrices


# def getV(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, mathcal_N, V):
#     v = collections.defaultdict(dict)
#     for pair in mathcal_N:
#         v[ pair[0] ][ pair[1] ]  = getVij(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, V,  pair[0], pair[1])
#     return v
  
# def getVij(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, V, i1, i2):

#     l1 = LossMatrix[i1]
#     l2 = LossMatrix[i2]
#     ldiff = l1 - l2

#     m = gp.Model( )
#     m.Params.LogToConsole = 0

#     vars = collections.defaultdict(dict)
#     for k in V[i1][i2] :
#         vars[k] = []
#         sk = len( set(FeedbackMatrix[k]) )
#         for a in range( sk ):
#             varName = "v_{}_{}_{}".format(i1, i2, a) 
#             vars[k].append( m.addVar(-GRB.INFINITY, GRB.INFINITY, 0., GRB.CONTINUOUS, varName ) ) 
#             m.update()

#     obj = 0
#     for k in  V[i1][i2] :
#         sk = len( set(FeedbackMatrix[k]) )
#         for a in range( sk ):
#             obj += vars[k][a]**2
#     m.setObjective(obj, GRB.MINIMIZE)

#     expression = 0
#     for k in  V[i1][i2] :
#         # print('signal', SignalMatrices[k].shape,'vars', vars[k] )
#         expression += SignalMatrices[k].T @ vars[k]
#     for l in range(len(ldiff)):
#         # print( ldiff[l],  )
#         m.addConstr( expression[l] == ldiff[l],  'constraint{}'.format(l) )

#     m.optimize()
    
#     vij = {}
#     for k in V[i1][i2] :
#         sk = len( set(FeedbackMatrix[k]) )
#         vijk = np.zeros( sk )
#         for a in range( sk ):
#             vijk[a] =   vars[k][a].X
#         vij[k] = vijk

#     return vij


# def getConfidenceWidth( mathcal_N, V, v,  N):
#     W = np.zeros(N)

#     for pair in mathcal_N:
#         for k in V[ pair[0] ][ pair[1] ]:
#             vec = v[ pair[0] ][ pair[1] ][k]
#             W[k] = np.max( [ W[k], np.linalg.norm(vec, np.inf) ] )
#     return W

  
# def f(t, alpha):
#     return   (t**(2/3) ) * ( alpha * np.log(t) )**(1/3)

  
        
        

