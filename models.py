import numpy as np


class KMeans:
    """
    Our Implementation of K-Means algorithm
    """
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001):
        """Initialize with K"""
        self.K = n_clusters
        self.mu = None
        self.labels_ = None
        self._Z = None
        self._max_iter = max_iter
        self._tol = tol
        # TODO: check parameters for validity
        self._is_trained = False
        
    def fit(self, data):
        """Trains the model to find optimal wieght and bias parameters"""
        assert (isinstance(data, np.ndarray) & (np.ndim(data) == 2)), \
                    "The training data has to be a two dimensional array"
        # get number of data points and dimension
        n, d = data.shape
        # initialize
        self.mu = data[np.random.choice(range(data.shape[0]), self.K, replace=False)]
        Z_old = np.zeros((n, self.K), dtype=int)
        l = 0
        while True:
            self._Z = np.zeros((n, self.K), dtype=int)
            for i in range(n):
                j = np.argmin(np.sum((self.mu - data[i]) ** 2, axis=1))
                self._Z[i, j] = 1
            # convergence
            if np.all(Z_old == self._Z) | (l >= self._max_iter):
                self._is_trained = True
                self.labels_ = np.where(self._Z)[1]
                break
            # if not update mu and continue
            for j in range(self.K):
                self.mu[j] = np.sum(self._Z[:, j].reshape(-1, 1) * data, axis=0) / np.sum(self._Z[:, j])
            Z_old = self._Z
            l += 1
    
    def distortion(self, data):
        assert self._is_trained, "Model not yet trained!"
        d = 0
        for i in range(self._Z.shape[0]):
            for j in range(self._Z.shape[1]):
                d += self._Z[i,j] * np.sum((data[i] - self.mu[j]) ** 2)
        return d
    
    def Z(self):
        return self._Z


class IGMM:
    """Our implementation of the Isotropic GMM Model"""
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001):
        """Initialize with K"""
        self.K = n_clusters
        self.pi = None
        self.mu = None
        self.var = None
        self.tau = None
        self.labels_ = None
        self._max_iter = max_iter
        self._tol = tol
        # TODO: check parameters for validity
        self._is_trained = False

    def _E(self, data):
        """E update"""
        n, d = data.shape
        T = np.zeros((n, self.K))
        for i in range(n):
            for j in range(self.K):
                T[i, j] = self.pi[j] * ((2 * np.pi * self.var[j]) ** (-d/2)) \
                           * np.exp(-np.sum((data[i] - self.mu[j]) ** 2) / (2 * self.var[j]))
        row_sums = T.sum(axis=1)
        self.tau = T / row_sums[:, np.newaxis]

    def _M(self, data):
        """M Update"""
        n, d = data.shape
        self.pi = np.mean(self.tau, axis=0)
        self.mu = (self.tau.T @ data) * (1 / np.sum(self.tau, axis=0).reshape(-1,1))
        for j in range(self.K):
            self.var[j] = np.dot(self.tau[:, j], np.sum((data - self.mu[j]) ** 2, axis=1)) \
                                                            / (d * np.sum(self.tau[:, j]))
    
    def _is_close(self, old_params):
        """Checks for convergence"""
        old_pi, old_mu, old_var = old_params
        return np.all(np.abs(self.pi - old_pi) < self._tol) \
                and np.all(np.abs(self.mu - old_mu) < self._tol) \
                and np.all(np.abs(self.var - old_var) < self._tol)
    
    def fit(self, data):
        """EM algorithm"""
        assert (isinstance(data, np.ndarray) & (np.ndim(data) == 2)), \
                    "The training data has to be a two dimensional array"
        n, d = data.shape
        km = KMeans(self.K, self._max_iter, self._tol)
        km.fit(data)
        self.mu = km.mu
        self.pi = np.sum(km.Z(), axis=0)/len(km.labels_)
        self.var = np.random.uniform(1, 5, self.K)
        old_pi = np.ones(self.K)
        old_mu = np.ones(self.mu.shape)
        old_var = np.ones(len(self.var.shape))
        l = 0
        while True:
            self._E(data)
            self._M(data)
            if self._is_close((old_pi, old_mu, old_var)) | (l >= self._max_iter):
                self._is_trained = True
                self.labels_ = np.argmax(self.tau, axis=1)
                print('EM updates converged at iteration', l)
                break
            old_pi, old_mu, old_var = self.pi, self.mu, self.var
            l += 1

    def predict(self, data):
        """Predicts labels for new data points"""
        n, d = data.shape
        T = np.zeros((n, self.K))
        for i in range(n):
            for j in range(self.K):
                T[i, j] = self.pi[j] * ((2 * np.pi * self.var[j]) ** (-d/2)) \
                           * np.exp(-np.sum((data[i] - self.mu[j]) ** 2) / (2 * self.var[j]))
        row_sums = T.sum(axis=1)
        return np.argmax(T / row_sums[:, np.newaxis], axis=1)

    def neg_log_llhd(self, data):
        """Predicts labels for new data points"""
        n, d = data.shape
        l = np.zeros(n)
        for i in range(n):
            for j in range(self.K):
                l[i] += self.pi[j] * ((2 * np.pi * self.var[j]) ** (-d/2)) \
                           * np.exp(-np.sum((data[i] - self.mu[j]) ** 2) / (2 * self.var[j]))
        return -np.sum(np.log(l))/n

class GMM:
    """Our implementation of the GMM Model"""
    def __init__(self, n_clusters=8, max_iter=300, tol=0.001):
        """Initialize with K"""
        self.K = n_clusters
        self.pi = None
        self.mu = None
        self.Sigma = None
        self.tau = None
        self.labels_ = None
        self._max_iter = max_iter
        self._tol = tol
        # TODO: check parameters for validity
        self._is_trained = False

    def _E(self, data):
        """E update"""
        n, d = data.shape
        T = np.zeros((n, self.K))
        for i in range(n):
            for j in range(self.K):
                det_S = np.linalg.det(self.Sigma[j])
                inv_S = np.linalg.inv(self.Sigma[j])
                T[i, j] = self.pi[j] * ((2 * np.pi) ** (-d/2)) * (det_S ** (-0.5)) \
                           * np.exp(-0.5 * ((data[i] - self.mu[j]).reshape(1, -1) 
                                            @  inv_S
                                            @ (data[i] - self.mu[j]).reshape(-1, 1)).item())
        row_sums = T.sum(axis=1)
        self.tau = T / row_sums[:, np.newaxis]

    def _M(self, data):
        """M Update"""
        n, d = data.shape
        self.pi = np.mean(self.tau, axis=0)
        self.mu = (self.tau.T @ data) * (1 / np.sum(self.tau, axis=0).reshape(-1,1))
        for j in range(self.Sigma.shape[0]):
            self.Sigma[j] = sum([self.tau[i, j] * ((data[i] - self.mu[j]).reshape(-1, 1) 
                                                   @ (data[i] - self.mu[j]).reshape(1, -1))
                                 for i in range(n)]) / np.sum(self.tau[:, j])
    
    def _is_close(self, old_params):
        """Checks for convergence"""
        old_pi, old_mu, old_Sigma = old_params
        return np.all(np.abs(self.pi - old_pi) < self._tol) \
                and np.all(np.abs(self.mu - old_mu) < self._tol) \
                and np.all(np.abs(self.Sigma - old_Sigma) < self._tol)
    
    def fit(self, data):
        """EM algorithm"""
        assert (isinstance(data, np.ndarray) & (np.ndim(data) == 2)), \
                    "The training data has to be a two dimensional array"
        n, d = data.shape
        # initialize
        km = KMeans(self.K, self._max_iter, self._tol)
        km.fit(data)
        self.mu = km.mu
        self.pi = np.sum(km.Z(), axis=0)/len(km.labels_)
        self.Sigma = np.array([s * np.eye(d) for s in np.random.uniform(1, 5, self.K)])
        # choose values for the previous step
        old_pi = np.ones(self.K)
        old_mu = np.ones(self.mu.shape)
        old_Sigma = np.ones(self.Sigma.shape)
        l = 0
        while True:
            self._E(data)  # E Step
            self._M(data)  # M Step
            if self._is_close((old_pi, old_mu, old_Sigma)) | (l >= self._max_iter):
                self._is_trained = True
                self.labels_ = np.argmax(self.tau, axis=1)
                print('EM updates converged at iteration', l)
                break
            old_pi, old_mu, old_Sigma = self.pi, self.mu, self.Sigma
            l += 1

    def predict(self, data):
        """Predicts labels for new data points"""
        n, d = data.shape
        T = np.zeros((n, self.K))
        for i in range(n):
            for j in range(self.K):
                det_S = np.linalg.det(self.Sigma[j])
                inv_S = np.linalg.inv(self.Sigma[j])
                T[i, j] = self.pi[j] * ((2 * np.pi) ** (-d/2)) * (det_S ** (-0.5)) \
                           * np.exp(-0.5 * ((data[i] - self.mu[j]).reshape(1, -1) 
                                            @  inv_S
                                            @ (data[i] - self.mu[j]).reshape(-1, 1)).item())
        row_sums = T.sum(axis=1)
        return np.argmax(T / row_sums[:, np.newaxis], axis=1)
    
    def neg_log_llhd(self, data):
        """Finds the mean negative log likelihood"""
        n, d = data.shape
        T = np.zeros((n, self.K))
        l = np.zeros(n)
        for i in range(n):
            for j in range(self.K):
                det_S = np.linalg.det(self.Sigma[j])
                inv_S = np.linalg.inv(self.Sigma[j])
                l[i] += self.pi[j] * ((2 * np.pi) ** (-d/2)) * (det_S ** (-0.5)) \
                           * np.exp(-0.5 * ((data[i] - self.mu[j]).reshape(1, -1) 
                                            @  inv_S
                                            @ (data[i] - self.mu[j]).reshape(-1, 1)).item())
        return -np.sum(np.log(l))/n