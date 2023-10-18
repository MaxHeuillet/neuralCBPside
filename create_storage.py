import gzip
import pickle as pkl
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--case", required=True, help="case")
parser.add_argument("--horizon", required=True, help="horizon of each realization of the experiment")
parser.add_argument("--n_folds", required=True, help="number of folds")
parser.add_argument("--game", required=True, help="game")
parser.add_argument("--context_type", required=True, help="context type")
parser.add_argument("--approach", required=True, help="algorithme")

args = parser.parse_args()

horizon = int(args.horizon)
n_folds = int(args.n_folds)

with gzip.open( './results/{}_{}_{}_{}_{}.pkl.gz'.format(args.case, args.game,  args.context_type, horizon, n_folds, args.approach) ,'wb') as g:
    pkl.dump( [None]*horizon, g)