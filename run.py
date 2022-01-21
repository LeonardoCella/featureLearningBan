from featLearnBan.policies.SA_FeatLearnBan import SA_FeatLearnBan
from featLearnBan.policies.Random import Random
from featLearnBan.policies.Oful import Oful
from featLearnBan.policies.Lasso import Lasso
from featLearnBan.environment import MTL
from featLearnBan.Evaluation import Evaluation

from numpy import arange, cumsum, log, zeros
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import rc


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
VERBOSE = True

parser = OptionParser("Usage: %prog [options]", version="%prog 1.0")
parser.add_option('-d', dest = 'd', default = "20", type = 'int', help = "Number of features")
parser.add_option('-K', dest = 'K', default = "10", type = 'int', help = "Number of arms")
parser.add_option('-T', dest = 'T', default = "20", type = 'int', help = "Number of rounds")
parser.add_option('-N', dest = 'N', default = "30", type = 'int', help = "Number of taks")
parser.add_option('--shU', dest = 'shU', default = "10", type = 'int', help = "Shrinking Users")
parser.add_option('--shI', dest = 'shI', default = "10", type = 'int', help = "Shrinking Items")
parser.add_option('--nrep', dest = 'nrep', default = "1", type = 'int', help = "Number Repetitions")


(opts, args) = parser.parse_args()
K = opts.K
T = opts.T
N_TASK = opts.N
d = opts.d
shU = opts.shU
shI = opts.shI
nrep = opts.nrep

assert K > 0, "Not consistent arms number parameter"
assert nrep > 0, "Not consistent number of repetitions"
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
policies["SA FeatLearnBan 0"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[0])] 
policies_name.append("SA FeatLearnBan 0")
policies["SA FeatLearnBan 1"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[1])] 
policies_name.append("SA FeatLearnBan 1")
policies["SA FeatLearnBan 10"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[2])] 
policies_name.append("SA FeatLearnBan 10")
policies["OFUL"] = [Oful(T, d, K, reg[0]) for _ in range(N_TASK)]
policies_name.append("OFUL")
policies["LASSO"] = [Lasso(T, d, K, reg[0]) for _ in range(N_TASK)]
policies_name.append("LASSO")


assert len(policies_name) == len(policies), "Inconsistent Policies"
assert N_TASK >= nrep, "This is required to have a consistent seed"

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
    # Here each policy p is a list of N_TASK policies except for SA_FeatLearnBan
    mtl = MTL(DATA, N_TASK, T, K, d, s0, variance, p, p_name, noisy_rewards, shU, shI)
    evaluation = Evaluation(mtl, T, p_name, nrep, N_TASK)
    results.append(evaluation.getResults())  # Result Class: (Policy Name, Mean Rewards, Std Rewards)

if VERBOSE:
    # ===
    # RESULTS VISUALIZATION
    # ===
    fig = plt.figure(1)
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
    plt.legend(loc=2)
    plt.savefig(
        'output/full_cmpSA_featLearn{}_d{}_s{}_T{}_rep{}_tasks{}_arms{}_noisy{}_shU{}_shI{}.png'.format(DATA, d, s0, T, nrep, N_TASK, K, noisy_rewards, shU, shI))
