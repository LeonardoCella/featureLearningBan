from featLearnBan.Results import Result
from featLearnBan.environment.Environment import Environment
from featLearnBan.arms.LinearArm import LinearArm
from numpy import all, arange, eye, ones, random, sort, transpose, where, zeros


class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""

    def __init__(self, N, T, K, d, s0, variance, p_name, noisy_rewards):
        print("MAB_INIT: Num task {}, Horizon {}, Arms {}, d {}, Policy {}".format(N, T, K, d, p_name))
        self._N_TASKS = N
        self._T = T
        self._K = K
        self._arm_indexes = arange(self._K)
        self._d = d
        self._s0 = s0
        self._p_name = p_name
        self._variance = variance
        self._noisy_rewards = noisy_rewards

    def play(self, policy, horizon, nb_repetition):
        ''' Called once per policy from __init__ of Evaluation. Rounds scheduler.'''
        # print("MAB.py, play() {}".format(nb_repetition))
        assert horizon == self._T, "Inconsistent parameters passing."

        # Result data structure initialization
        result = Result(horizon)
        t = 0  # round counter

        for nb_task in range(self._N_TASKS):
            # As tasks are independent we need to reset each policy
            policy.reset()

            # Linear Task Specification (nb_repetition)
            random.seed(nb_repetition + nb_task * self._N_TASKS)
            wstar = zeros(self._d)
            binary_mask = sort(random.choice(range(self._d), self._s0, replace=False))
            wstar[binary_mask] = random.uniform(low=0., high=1., size=self._s0)

            # Linear Arm Initialization (Linear Reward)
            linear_arm = LinearArm(wstar)

            # T-dimensional noise vector (nb_repetition)
            noise = self._variance * random.randn(horizon)

            # Sequence of interactions
            while t < horizon:
                # print("\n===\nMAB.py, play(): round {}\n===".format(t))

                if t == 0:
                    # print("MAB.py, play(): current pull counters: {}".format(self._arms_pull_counter))

                    # Initialization Optimal Policy based on considered instance
                    if self._p_name == "Optimal":
                        policy.initialize(wstar)

                # Arm-noise Creation
                X = self._arm_creation()  # K x d

                # Chosen arm (arm index)
                if self._p_name == "Oracle":
                    # Create subvector
                    X_sparse = X[:, binary_mask]
                    choice = policy.choice(t, X_sparse)
                else:
                    choice = policy.choice(t, X)


                expected_reward = linear_arm.draw(X[choice])
                noisy_reward = expected_reward + noise[t]

                # Reward with (possible) cost penalization
                if self._noisy_rewards:
                    result.store(t, choice, noisy_reward)
                    if self._p_name == "Oracle":
                        policy.update(t, X_sparse[choice], noisy_reward)
                    else:
                        policy.update(t, X[choice], noisy_reward)
                else:
                    result.store(t, choice, expected_reward)
                    if self._p_name == "Oracle":
                        policy.update(t, X_sparse[choice], expected_reward)
                    else:
                        policy.update(t, X[choice], expected_reward)

                # print("Chosen arm at {}, by {}, {} with rwd {}".format(t, self._p_name, choice, noisy_reward))

                t = t + 1

        return result

    def _arm_creation(self):
        sigma_sq = 1.
        rho_sq = 0.7
        V = (sigma_sq - rho_sq) * eye(self._K) + rho_sq * ones((self._K, self._K))
        x = random.multivariate_normal(zeros(self._K), V, self._d).T  # 0 mean K dim, covariance V KxK, d samples
        # x_stack = x.reshape(self._K * self._d)
        # print("MAB arms_creation() arms shape {}, value {}".format(x.shape, x[:2, :2]))
        return x
