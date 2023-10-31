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
############################################# PULP:
##########################################################################################

def solve_LP(args):
    z, M, LossMatrix, halfspace = args
    vars = [lp.LpVariable('p_{}'.format(i), 0.00001, 1.0, lp.LpContinuous) for i in range(M)]
    m = lp.LpProblem(name="Pareto_Optimization_{}".format(z), sense=lp.LpMinimize)
    m += (lp.lpSum(vars) == 1.0, "css")
    
    lossExprs = np.dot(LossMatrix - LossMatrix[z, :], vars)
    for i2, lossExpr in enumerate(lossExprs):
        if i2 != z:
            m += (lossExpr >= 0.0, 'c_{}'.format(i2))
    
    for element in halfspace:
        pair, sign = element
        if sign != 0:
            halfspaceExpr = lp.lpSum(sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M))
            m += (halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))
    
    m.solve()
    if lp.LpStatus[m.status] == "Optimal":
        return z
    return None



def getParetoOptimalActions(LossMatrix, N, M, halfspace, num_pools=None):
    lp.LpSolverDefault.msg = 0
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
    lp.LpSolverDefault.msg = 0  # or False

    m = lp.LpProblem(name="Neighbor_Check", sense=lp.LpMinimize)

    # Define variables
    vars = [lp.LpVariable("p_{}".format(j), 0.00001, 1.0, lp.LpContinuous) for j in range(M)]

    # Add simplex constraint
    m += (lp.lpSum(vars) == 1.0, "css")

    # Add two-degenerate constraint
    twoDegenerateExpr = lp.lpSum((LossMatrix[i2][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
    m += (twoDegenerateExpr == 0.0, "cdeg")

    # Add loss constraints
    for i3 in range(N):
        if i3 != i1:
            lossExpr = lp.lpSum((LossMatrix[i3][j] - LossMatrix[i1][j]) * vars[j] for j in range(M))
            m += (lossExpr >= 0.0, "c_{}".format(i3))

    # Add halfspace constraints
    for element in halfspace:
        pair, sign = element[0], element[1]
        if sign != 0:
            halfspaceExpr = lp.lpSum(
                sign * (LossMatrix[pair[0]][j] - LossMatrix[pair[1]][j]) * vars[j] for j in range(M)
            )
            m += (halfspaceExpr >= 0.001, "ch_{}_{}".format(pair[0], pair[1]))

    m.solve()
    return lp.LpStatus[m.status] == "Optimal"


        

