# importing packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import pickle
from sklearn import manifold

# function to producd a manifold from a corr matrix
def produce_manifold(curr_corr_matrix, session_info, num_dim):
    iso = manifold.Isomap(n_neighbors=len(curr_corr_matrix) - 1, n_components=num_dim)
    print("Fitting Manifold...")
    iso.fit(curr_corr_matrix)
    print("Transforming Manifold...")
    manifold_xD_trans = iso.transform(curr_corr_matrix)
    # Left with x dimensions
    col = []
    for i in range(num_dim):
        col.append("Component {}".format(i+1))
    manifold_xD = pd.DataFrame(manifold_xD_trans, columns=col)

    print("Saving Manifold...")
    # save the original data frame
    with open('manifold_{}D_on_corr_{}.pickle'.format(num_dim, session_info), 'wb') as f:
        pickle.dump(manifold_xD, f)

# Load in the data
session13 = np.load("../Data/Session13/sessionM409828_13_CALCIUM-FLUORESCENCE.npy")

# get a list of all the ns for the column names
ns = []
for i in range(session13[0,:].shape[0]):
    ns.append("n{}".format(i))

# Make a dataframe of the data for easier processing
session13_df = pd.DataFrame(session13, columns = ns)

print("Producing Correlation Matrix...")
# Build a correlation matrix of the data, then produce a manifold
corr_session13_df= pd.DataFrame(np.corrcoef(session13_df.values, rowvar=False), columns=session13_df.columns)
print("Produced Correlation Matrix Successfully")
corr_matrix = 1 - corr_session13_df
print("Producing Manifold...")
produce_manifold(corr_matrix, "V1DD_Session13", 2)
print("Written Manifold!")
