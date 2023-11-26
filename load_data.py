from sklearn.datasets import fetch_openml
import pickle as pkl
import gzip

# Downloading the dataset
X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)

# Saving both X and y to a single Pickle file
with open('./data/MagicTelescope.pkl', 'wb') as file:
    pkl.dump((X, y), file)


X, y = fetch_openml('adult', version=2, return_X_y=True)

with open('./data/adult.pkl', 'wb') as file:
    pkl.dump((X, y), file)