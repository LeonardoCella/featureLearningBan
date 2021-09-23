from numpy import zeros, transpose, sqrt, outer, matrix, log, eye, dot, asarray, argmax
from numpy.linalg import norm, det
from featLearnBan.policies.Policy import Policy
from featLearnBan.tools.matrix_functions import sherman_morrison


class Oful(Policy):
    def __init__(self, horizon, d, nb_arms, reg):
        # print("OFUL Rounds {}, Dimension {}, arms {}".format(horizon, d, nb_arms)) 
        self._T = horizon
        self._n_arms = nb_arms
        self._d = d
        self._reg = reg
        self._w_hat = zeros(self._d) 
        self._beta = 0
        self._Ireg = eye(self._d) * reg
        self._A_inv = eye(self._d) * (1.0/reg)
        self._A = eye(self._d) * reg
        self._b = zeros(self._d)
        self._t = 0

    def update(self, j, t, arm, reward):
        self._A_inv = sherman_morrison(self._A_inv, arm)
        self._A += dot(arm, transpose(arm))
        self._b += arm*reward
        self._w_hat = dot(self._A_inv, self._b)
        beta = sqrt(2*log(det(self._A)/det(self._Ireg)))
        self._beta = beta

    def choice(self, j, t, context_vectors):
        self._t = t + 1
        pred_values = []
        wtx_array = dot(context_vectors, self._w_hat).T
        for a, x in enumerate(context_vectors):
            # Confidence Bound wrt direction x
            cb = self._beta * (
                (dot(transpose(x), dot(self._A_inv, x)) * log(1 + t)) ** (0.5))
            # OFU estimates
            pred_value = wtx_array[a] + cb
            pred_values.append(pred_value)
        # argmax over OFU estimates
        max_index = pred_values.index(max(pred_values))
        #print("Arms {}".format(context_vectors))
        #print("w_star {}".format(self._w_star))
        #print("w_hat {}".format(self._w_hat))
        #print("PRED {}".format(wtx_array))
        #print("UCB {}".format(pred_values))

        return max_index


    def reset(self):
        self._w_hat = zeros(self._d)
        self._beta = 0
        self._A_inv = eye(self._d) * (1/self._reg)
        self._A = eye(self._d) * self._reg
        self._b = zeros(self._d)
        return

    def getEstimate(self):
        return self._w_hat
