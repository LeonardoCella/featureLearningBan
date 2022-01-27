from featLearnBan.Results import Result
from featLearnBan.environment.Environment import Environment
from featLearnBan.arms.LinearArm import LinearArm
from featLearnBan.arms.LastFM import LastFM
from featLearnBan.arms.MovieLens import Movielens

from numpy import all, arange, argmax, concatenate, eye, ones, random, sort, transpose, zeros
from scipy import io as sio
import sys
import random as rnd


class MTL(Environment):
    """MTL multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, DATA, N, T, K, d, s0, variance, policies, p_name, noisy_rewards, shU = 10, shI = 10):
        print("MTL_INIT: Num task {}, Horizon {}, Arms {}, d {}, Policy {}".format(N, T, K, d, p_name))
        self._DATA = DATA
        self._N_TASKS = N
        self._T = T  # refers to a single task
        self._K = K
        self._arm_indexes = arange(self._K)
        self._d = d
        self._s0 = s0
        self._policies = policies  # List of ntasks independent policies or single MTL policy
        self._p_name = p_name
        self._variance = variance
        self._noisy_rewards = noisy_rewards
        self._shU = shU
        self._shI = shI

    def play(self, horizon, nbTasks, nb_repetition):
        ''' Called once per policy from __init__ of Evaluation. Rounds scheduler.'''

        assert horizon == self._T, "Inconsistent parameters passing."
        assert nbTasks == self._N_TASKS, "Inconsistent parameters passing."
        print("MTL play()")

        # Result data structure initialization
        result = Result(horizon * nbTasks)
        t = 0  # round counter

        # Reset the policies during different repetitions
        for p in self._policies:
            p.reset()

        task_parameters = zeros((self._N_TASKS, self._d))

        # SEED
        random.seed(nb_repetition)  # SAME MASK OVERALL TASKS/POLICIES

        print("MTL Data Extraction")
        if self._DATA == 0:  # Synthetic Data

            binary_mask_vectors = sort(random.choice(range(self._d), self._s0, replace=False))
            for j in range(self._N_TASKS):
                task_parameters[j, binary_mask_vectors] = random.uniform(low=0., high=1.,
                                                                         size=self._s0)  # TASK VECTORS CREATION

            # LINEAR ARMS INITIALIZATION (Linear Reward) - one per task
            linear_arm_functions = [LinearArm(wstar) for wstar in task_parameters]
            # print("W vectors: ", task_parameters)

            # N_TASKS T-dimensional noise vector (nb_repetition)
            noise_vectors = [self._variance * random.randn(horizon) for _ in range(self._N_TASKS)]
            # print("NOISE", noise_vectors)

        else:  # Lenk or LastFM or Movielens data
            if self._DATA == 1:
                # Extract contexts and rwds listed by task
                contexts_list_bytask, rwds_list_bytask = self._lenk_preprocessing()
            else:
                if self._DATA == 2:  # LastFM
                    # Extract contexts and rwds listed by task
                    print("MTL LastFM")
                    lastFM = LastFM(self._N_TASKS, self._d, self._K, self._T, nb_repetition, self._shU, self._shI)
                else:  # Movielens
                    movielens = Movielens(self._N_TASKS, self._d, self._K, self._T, self._shU, self._shI)


        # Sequence of interactions
        sequential_round = 0  # keeps the counter across tasks, used by Eval.store()
        while t < horizon:

            for j in range(self._N_TASKS):  # Iterates over policies, task vectors, arm reward functions

                # DATA PREPARATION
                if self._DATA == 0:  # SYNTH.
                    # Single Task Parameters
                    wstar = task_parameters[j]
                    linear_arm = linear_arm_functions[j]
                    noise = noise_vectors[j]
                else: # SINGLE TASK-ROUND LENK/LASTFM/MOVIELENS DATA
                    if self._DATA == 1:  # LENK
                        context_single_task = contexts_list_bytask[j]
                        rwd_single_task = rwds_list_bytask[j]
                        sample_indexes = rnd.choices(range(context_single_task.shape[0]),
                                                 k=self._K)  # the adopted rnd is not the numpy one
                    else:
                        if self._DATA == 2:  # LASTFM
                            context_single_task = lastFM.get_context(j, t)
                            rwd_single_task = lastFM.get_rwd(j, t)
                        else:  # MOVIELENS
                            context_single_task = movielens.get_context(j, t)
                            rwd_single_task = movielens.get_rwd(j, t)

                if len(self._policies) > 1:  # Independent policies, not MTL learners
                    policy = self._policies[j]
                else:  # MTL 
                    policy = self._policies[0]

                opt_arm = 0
                # Initialization Optimal Policy based on considered instance
                if self._p_name == "Optimal":
                    if self._DATA == 0 and t == 0:
                        policy.initialize(wstar)

                # Independently on the optimal policy, we need to compute the optimal arm
                if self._DATA == 1:  # Lenk Data
                    if self._noisy_rewards:  # Noisy version
                        noise = random.normal(0, self._variance, self._K)
                        opt_arm = argmax(rwd_single_task[sample_indexes] + noise)
                    else:  # No noise
                        opt_arm = argmax(rwd_single_task[sample_indexes])
                else:  # LastFM or Movielens Data
                    if self._noisy_rewards:  # Noisy version
                        noise = random.normal(0, self._variance, self._K)
                        opt_arm = argmax(rwd_single_task + noise)
                    else:  # No noise
                        opt_arm = argmax(rwd_single_task)

                # Contexts Creation
                if self._DATA == 0:
                    X = self._arm_creation()  # K x d
                else:  # Lenk/LastFM/Movielens
                    if self._DATA == 1:  # Lenk Data
                        X = context_single_task[sample_indexes]
                    else:  # LastFM/Movielens Data
                        X = context_single_task

                # CHOICE
                if self._p_name == "Oracle":
                    assert self._DATA > 0, "Inconsistent loaded policy!"
                    # Create subvector
                    X_sparse = X[:, binary_mask_vectors]
                    choice = policy.choice(j, t, X_sparse)
                else:
                    choice = policy.choice(j, t, X)

                # FEEDBACK
                if self._DATA == 0:
                    expected_reward = linear_arm.draw(X[choice])
                    noisy_reward = expected_reward + noise[t]
                    if self._noisy_rewards:
                         expected_reward = noisy_reward
                else:  # Lenk, LastFM, Movielens Data
                    expected_reward = 0
                    if choice == opt_arm:
                        expected_reward = expected_reward + 1
                    noisy_reward = expected_reward
                

                # UPDATES (SO FAR ONLY BASED ON EXPECTED REWARD)
                # Reward with (possible) cost penalization
                #if self._noisy_rewards:
                #    result.store(sequential_round, choice, noisy_reward)
                #    if self._p_name == "Oracle":
                #        policy.update(j, t, X_sparse[choice], noisy_reward)
                #    else:
                #        policy.update(j, t, X[choice], noisy_reward)
                    # print("MTL j {}, t {}, arm {}, rwd {}".format(j, t, choice, noisy_reward))
                #else:
                result.store(sequential_round, choice, expected_reward)
                if self._p_name == "Oracle":
                    assert self._DATA == 0, "Inconsistent protocol in MTL.py"
                    policy.update(j, t, X_sparse[choice], expected_reward)
                else:
                    if self._DATA == 0:
                        policy.update(j, t, X[choice], expected_reward)
                    else:  # Lenk/LastFM/Movielens Data
                        policy.update(j, t, context_single_task[choice, :], expected_reward)
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

    def _lenk_preprocessing(self):
        lenk = sio.loadmat('featLearnBan/arms/lenk_data.mat')
        train = lenk['Traindata']
        test = lenk['Testdata']
        tr_contexts, te_contexts = train[:, 5:14], test[:, 5:14]
        tr_rwds, te_rwds = train[:, -1], test[:, -1]

        def split_tasks(data, nt, number_of_elements):
            return [data[i * number_of_elements:(i + 1) * number_of_elements] for i in range(nt)]

        n_tasks = self._N_TASKS
        ne_tr = 16  # number of elements on train set per task
        ne_test = 4  # number of elements on test set per task

        contexts_tr_m = split_tasks(tr_contexts, n_tasks, ne_tr)
        rwds_tr_m = split_tasks(tr_rwds, n_tasks, ne_tr)

        contexts_te_m = split_tasks(te_contexts, n_tasks, ne_test)
        rwds_te_m = split_tasks(te_rwds, n_tasks, ne_test)

        # Task Aggregation
        task_contexts = [concatenate([contexts_tr_m[i], contexts_te_m[i]]) for i in range(n_tasks)]
        task_rwds = [concatenate([rwds_tr_m[i], rwds_te_m[i]]) for i in range(n_tasks)]

        return (task_contexts, task_rwds)

    def _movielens_ratings_preprocessing(self):
        '''Matrix shrinking'''

        with open("featLearnBan/arms/MovieLens/ratings.dat", 'r') as f:
            urm_byrecords = read_table(f, sep = "::", header = None, names = ['UserID', 'ItemID', 'rating', 'timestamp'], engine = 'python')
        urm_byrecords = urm_byrecords.drop(['timestamp'], axis = 1)
        # Filter Low Rated Movies
        urm_byrecords = urm_byrecords[urm_byrecords.groupby('ItemID')['ItemID'].transform('count').ge(self._shrinkItem)]
        # Filter Low Rater Users
        urm_byrecords = urm_byrecords[urm_byrecords.groupby('UserID')['UserID'].transform('count').ge(self._shrinkUser)]
        # Remove rows with NaN
        urm_byrecords.dropna(axis = 0, how = 'any', inplace = True)
        # Convert from Records to DF
        urm_df = urm_byrecords.pivot_table(index = "UserID", columns = "ItemID", values = "rating", fill_value = 0)
        # Storing
        self._urm_byrecords = urm_byrecords
        self._urm_df = urm_df
        print("URM shape {}".format(self._urm_df.shape))
        return
