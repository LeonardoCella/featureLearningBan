from featLearnBan.Results import Result
from featLearnBan.environment.Environment import Environment
from featLearnBan.arms.LinearArm import LinearArm
from featLearnBan.arms.LastFM import LastFM
from featLearnBan.arms.MovieLens import Movielens

from numpy import all, arange, argmax, concatenate, eye, ones, random, sort, transpose, zeros
from scipy import io as sio
import sys
import random as rnd


class MTL_ICML(Environment):
    """MTL multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, N, T, K, d, r, variance, policies, p_name, noisy_rewards):
        # print("MTL_INIT: Num task {}, Horizon {}, Arms {}, d {}, Policy {}".format(N, T, K, d, p_name))
        self._N_TASKS = N
        self._T = T  # refers to a single task
        self._K = K
        self._arm_indexes = arange(self._K)
        self._d = d
        self._r = r
        self._policies = policies  # List of ntasks independent policies or single MTL policy
        self._p_name = p_name
        self._variance = variance
        self._noisy_rewards = noisy_rewards

    def play(self, horizon, nbTasks, nb_repetition):
        ''' Called once per policy from __init__ of Evaluation. Rounds scheduler.'''

        assert horizon == self._T, "Inconsistent parameters passing."
        assert nbTasks == self._N_TASKS, "Inconsistent parameters passing."
        #print("MTL play()")

        # Result data structure initialization
        result = Result(horizon * nbTasks)
        t = 0  # round counter

        # Reset the policies during different repetitions
        for p in self._policies:
            p.reset()

        task_parameters = zeros((self._N_TASKS, self._d))

        # SEED
        random.seed(nb_repetition)  # SAME MASK OVERALL TASKS/POLICIES

        binary_mask_vectors = sort(random.choice(range(self._d), self._r, replace=False))
        for j in range(self._N_TASKS):
            task_parameters[j, binary_mask_vectors] = random.uniform(low=0., high=1.,
                                                                     size=self._r)  # TASK VECTORS CREATION

        # LINEAR ARMS INITIALIZATION (Linear Reward) - one per task
        linear_arm_functions = [LinearArm(wstar) for wstar in task_parameters]
        # print("W vectors: ", task_parameters)


        # N_TASKS T-dimensional noise vector (nb_repetition)
        noise_vectors = [self._variance * random.randn(horizon) for _ in range(self._N_TASKS)]
        # print("NOISE", noise_vectors)


        # Sequence of interactions
        sequential_round = 0  # keeps the counter across tasks, used by Eval.store()
        while t < horizon:

            for j in range(self._N_TASKS):  # Iterates over policies, task vectors, arm reward functions

                # DATA PREPARATION
                # Single Task Parameters
                wstar = task_parameters[j]
                linear_arm = linear_arm_functions[j]
                noise = noise_vectors[j]
                assert len(noise) == horizon, "Inconsistent noise vector created."

                if len(self._policies) > 1:  # Independent policies, not MTL learners
                    policy = self._policies[j]
                else:  # MTL 
                    policy = self._policies[0]

                opt_arm = 0
                # Initialization Optimal Policy based on considered instance
                if self._p_name == "Optimal":
                    if t == 0:
                        policy.initialize(wstar)

                # Contexts Creation
                X = self._arm_creation()  # K x d

                # CHOICE
                if self._p_name == "Oracle":
                    X_sparse = X[:, binary_mask_vectors]
                    choice = policy.choice(j, t, X_sparse)
                else:
                    choice = policy.choice(j, t, X)

                # FEEDBACK
                expected_reward = linear_arm.draw(X[choice])
                noisy_reward = expected_reward + noise[t]

                result.store(sequential_round, choice, expected_reward)
                if self._p_name == "Oracle":
                    policy.update(j, t, X_sparse[choice], expected_reward)
                else:
                    policy.update(j, t, X[choice], expected_reward)
                    #print("MTL j {}, t {}, arm {}, rwd {}".format(j, t, choice, expected_reward))

                sequential_round = sequential_round + 1  # Across tasks

            t = t + 1  # within each task

        return result

    def _arm_creation(self):
        sigma_sq = 1.
        rho_sq = 0.7
        V = (sigma_sq - rho_sq) * eye(self._K) + rho_sq * ones((self._K, self._K))
        x = random.multivariate_normal(zeros(self._K), V, self._d).T  # 0 mean K dim, covariance V KxK, d samples
        return x

