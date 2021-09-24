#from featLearnBan.policies.RepLearning import RepLearning
from featLearnBan.policies.Random import Random
from featLearnBan.policies.Oful import Oful
from featLearnBan.policies.Lasso import Lasso
from featLearnBan.environment import MTL
from featLearnBan.Evaluation import Evaluation

from numpy import arange, cumsum, log, zeros
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'size': '23.0'})
rc('text', usetex=True)

# ===
# RUNNING PARAMETERS
# ===
SYNTH, LENK, LASTFM, MOVIE = 0, 1, 2, 3
DATA = LASTFM

d = [50, 9, 20, 20][DATA]  # Num Features
s0 = [5, 1, 5, 5][DATA]  # Rank(W) parameter
K = [50, 15, 10, 30][DATA]  # 100  # Num Arms
T = [50, 15, 30, 20][DATA]  # 200  # Horizon
N_TASK = [100, 50, 20, 40][DATA]  # 100  # Num Tasks
variance = [0.05, 1., 1., 1.][DATA]

noisy_rewards = [True, True, True, True][DATA]
N_REP = 2  # Number of Repetitions
VERBOSE = True

assert K > 0, "Not consistent arms number parameter"
assert d > s0, "Not consistent sparsity-features relation"
assert T > 0, "Not consistent horizon parameter"
assert N_TASK > 0, "Not consistent number of tasks"

if VERBOSE:
    print("RUN Horizon {}, Tasks {}, Arms {}, dimensions {}".format(T, N_TASK, K, d))

# ===
# INITIALIZATION
# ===
# Policies
policies = {}
policies_name = []

reg = [.1, 1, 10]

# Random Policy
#policies["Random"] = [Random(K) for _ in range(N_TASK)]
#policies_name.append("Random")
#policies["RepLearning"] = [RepLearning(N_TASK,T,K)] 
#policies.append("RepLearning")
policies["OFUL"] = [Oful(T, d, K, reg[0]) for _ in range(N_TASK)]
policies_name.append("OFUL")
policies["LASSO"] = [Lasso(T, d, K, reg[0]) for _ in range(N_TASK)]
policies_name.append("LASSO")


assert len(policies_name) == len(policies), "Inconsistent Policies"
assert N_TASK >= N_REP, "This is required to have a consistent seed"

# Plot parameters
colors = ['r', 'g', 'c', 'b', 'y', 'm', '#ae34eb', '#eb348f']
COLORS = {p_name: colors[i] for i, p_name in enumerate(policies_name)}
markers = ["o", "^", "v", "<", ">", ",", "h", "x"]
MARKERS = {p_name: markers[i] for i, p_name in enumerate(policies_name)}
linestyle = ["-", "-", "--", "-.", ":", "--", "-.", ":"]
LINESTYLE = {p_name: linestyle[i] for i, p_name in enumerate(policies_name)}

# ===
# RUN OVER POLICIES
# ===
results = []
test_results = []
for p_name, p in policies.items():
    print("\n=======NEW RUN=======")
    print("===POLICY {}===".format(p_name))

    # Here each policy p is a list of N_TASK policies except for RepLearning 
    mtl = MTL(DATA, N_TASK, T, K, d, s0, variance, p, p_name, noisy_rewards)
    evaluation = Evaluation(mtl, T, p_name, N_REP, N_TASK)
    results.append(evaluation.getResults())  # Result Class: (Policy Name, Mean Rewards, Std Rewards)

if VERBOSE:
    # ===
    # RESULTS VISUALIZATION
    # ===
    fig = plt.figure(1, figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Rewards Extraction
    POLICY_AVGS = []  # cum sum mean rwd
    POLICY_STD = []
    POLICY_N = []
    round_indexes = arange(T * N_TASK)

    for name, avg, std in results:  # list of (policy name, meanRwds, stdRwds)
        cumsumavg = cumsum(avg)
        plt.fill_between(round_indexes, cumsumavg - std / 2, cumsumavg + std / 2, alpha=0.5, color=COLORS[name])
        plt.plot(round_indexes, cumsumavg, color=COLORS[name], marker=MARKERS[name],
                 label="Policy {}".format(name))
        print("\nPOL {}, RWDS EVERY {}".format(name, [cumsumavg[i] for i in arange(0,len(cumsumavg),10)]))
        print("FINAL {}".format(cumsumavg[-1]))

    for j in range(N_TASK):
        plt.axvline(x=(j + 1) * T, color='b', linewidth=1.0, linestyle='--')

    ax.set_ylabel('Cumulative Rewards')
    ax.set_xlabel('Rounds across Tasks')
    ax.set_title('Multi-Tasking with {} tasks'.format(N_TASK))
    ax.yaxis.grid(True)

    # Save the figure and show
    # plt.tight_layout()
    plt.legend(loc=2)
    plt.savefig(
        'output/featLearn{}_d{}_s{}_T{}_rep{}_tasks{}_arms{}_noisy{}.png'.format(DATA, d, s0, T, N_REP, N_TASK, K, noisy_rewards))
    plt.show()
