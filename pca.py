import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = []
    
    def fit(self,X_train):
        X_train = (X_train - np.mean(X_train, axis=0) )#/ X_train.std()
        COV = np.cov(X_train.T)
        vals, vecs = np.linalg.eigh(COV)
        valvec = list(zip(vals, vecs.T))
        valvec.sort(key=lambda x:abs(x[0]), reverse=True)
        self.components = []
        if not self.n_components:
            self.components = [vec[1] for vec in valvec]
        else:
            self.components = [vec[1] for vec in valvec[:self.n_components]]