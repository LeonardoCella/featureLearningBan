'''Utility class for the performance evaluation'''

__version__ = "0.1"

from numpy import mean, std, zeros
from joblib import Parallel, delayed
import sys

sys.maxsize = 1000000  # Avoid truncations in print


def parallel_repetitions(evaluation, horizon, nbTasks, i):
    # print("parallel repetitions")
    result = evaluation.environment.play(horizon, nbTasks, i)
    return i, result


class Evaluation:
    def __init__(self, env, horizon, policyName, nbRepetitions, nbTasks):
        ''' Initialized in the run.py file.'''

        # Associated learning problem: policy, environment
        self.environment = env  # MAB instance

        # Learning problem parameters: horizon, policy name, nbRepetitions
        self.horizon = horizon
        self.polName = policyName
        self.nbRepetitions = nbRepetitions
        self.nbTasks = nbTasks

        # Data Structures to store the results of different reward samples
        self.rewards = zeros((self.nbRepetitions, self.horizon * self.nbTasks))  # nbRep x (T x nbTasks)

        # print("===Evaluation.py, INIT: {} over {} rounds for {} nbRepetitions".format(self.polName, self.horizon, self.nbRepetitions))

        # Parallel call to the policy run over the number of repetitions
        with Parallel(n_jobs=self.nbRepetitions, backend="multiprocessing") as parallel:
            print("Evaluation in WITH: Parallel")
            rep_results = parallel(delayed(parallel_repetitions)(self, self.horizon, self.nbTasks, i) for i in range(nbRepetitions))

        # Results extrapolation
        for i, result in rep_results:
            self.rewards[i] = result.getRewards()  # T instantaneous rewards

        # Averaged best Expectation
        self.meanRwd = mean(self.rewards, axis=0)  # (T,)
        self.stdRwd = std(self.rewards, axis=0)  # (T,)
        assert self.meanRwd.shape[0] == self.horizon * self.nbTasks, "Incoherent rewards"
        self.result = (policyName, self.meanRwd, self.stdRwd)

    def getResults(self):
        return self.result
