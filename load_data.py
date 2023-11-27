from sklearn.datasets import fetch_openml
import pickle as pkl
import gzip

print('download MagicTelescope')

# Downloading the dataset
X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)

print('save MagicTelescope')

# Saving both X and y to a single Pickle file
with open('./data/MagicTelescope.pkl', 'wb') as file:
    pkl.dump((X, y), file)



print(' download adult')

X, y = fetch_openml('adult', version=2, return_X_y=True)

print('save adult')

with open('./data/adult.pkl', 'wb') as file:
    pkl.dump((X, y), file)




print('download covertype')

X, y = fetch_openml('covertype', version=3, return_X_y=True)

print('save covertype')

with open('./data/covertype.pkl', 'wb') as file:
    pkl.dump((X, y), file)




print(' download shuttle')

X, y = fetch_openml('shuttle', version=1, return_X_y=True)

print('save shuttle')

with open('./data/shuttle.pkl', 'wb') as file:
    pkl.dump((X, y), file)
