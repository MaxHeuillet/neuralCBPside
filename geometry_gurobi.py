import numpy as np
import collections
import pulp as lp
from multiprocessing import Pool

def alphabet_size(FeedbackMatrix, N,M):
    alphabet = []
    for i in range(N):
        for j in range(M):
            alphabet.append(FeedbackMatrix[i][j])
    return len(set(alphabet)) 


##########################################################################################
############################################# GUROBI:
##########################################################################################



def solve_LP(args):
    z, M, LossMatrix, halfspace = args
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            #m.setParam('Threads', 1)
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