from numpy import arange, argmax, dot, expand_dims, hstack, log, maximum, mean, minimum, random, sqrt, zeros
from scipy.linalg import block_diag

from featLearnBan.policies.Policy import Policy
from . import traceNormRegression

class SA_FeatLearnBan(Policy): 
    def __init__(self, nb_tasks, horizon, d, nb_arms, reg):
        self._T = nb_tasks
        self._N = horizon
        self._k = nb_arms
        self._d = d
        self._reg = reg * sqrt(self._N/self._T) * log(self._T)
        self._X = [[] for _ in range(self._T)] # List of $T$ design matrices
        self._Y = [[] for _ in range(self._T)] # List of $T$ Target vectors
        self._What = zeros((self._d * self._T, 1)) # One column for each task vector

        self._iterations = 3000
        self._reg = reg 

    def reset(self):
        pass

    def choice(self, j, t, X):  # j:task, r:round, X: amrs as K*d matrix
        task = j
        first_delimiter = j * self._d
        second_delimiter = (j+1) * self._d
        w_t = self._What[first_delimiter:second_delimiter]
        wtx_array = dot(X, self._What[first_delimiter:second_delimiter]).T[0]
        max_index = list(wtx_array).index(max(wtx_array))
        self._X[task].append(X[max_index])
        #print("SA choice(): Task {}/{}, round {}/{}, values {}, max_i {}".format(j, self._T, t, self._N, wtx_array, max_index))
        return max_index

    def update(self, j, t, arm, rwd):
        #print("SA update(): Task {}/{}, round {}/{}, rwd {}".format(j, self._T, t, self._N, rwd))
        self._Y[j].append(rwd)
        if j == self._T - 1  and t > 0: # If last task but not the first round
            X_mtl = block_diag(*self._X)
            Y_mtl = hstack(self._Y).T
            Y_mtl = expand_dims(Y_mtl, axis = 1)
            initial = random.rand(self._d * self._T, 1)
            self._What, _ = traceNormRegression.accelerated_gradient_tracenorm(X_mtl, Y_mtl, self._reg, self._iterations, initial)


        
