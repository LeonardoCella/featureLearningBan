from numpy import zeros, transpose, sqrt, outer, matrix, log, eye, dot, asarray, argmax
from numpy.linalg import norm, det
from sklearn import linear_model
from featLearnBan.policies.Policy import Policy
from featLearnBan.tools.matrix_functions import sherman_morrison


class Lasso(Policy):
    def __init__(self, horizon, d, nb_arms, reg):
        self._T = horizon
        self._n_arms = nb_arms
        self._d = d
        self._reg = reg
        self._w_hat = zeros(self._d) 
        self._X = []
        self._Y = []
        self._t = 0

    def update(self, j, t, arm, reward):
        lambda_t = self._reg * sqrt(log(self._d)/(self._t))
        lasso = linear_model.Lasso(alpha = self._reg, max_iter = 5000, tol = 0.001)
        self._Y.append(reward)
        lasso.fit(self._X, self._Y)
        self._w_hat = lasso.coef_
        

    def choice(self, j, t, context_vectors):
        self._t = t + 1
        wtx_array = dot(context_vectors, self._w_hat).T
        max_index = list(wtx_array).index(max(wtx_array))
        self._X.append(context_vectors[max_index])

        return max_index


    def reset(self):
        self._w_hat = zeros(self._d)
        self._beta = 0
        self._b = zeros(self._d)
        return

    def getEstimate(self):
        return self._w_hat
