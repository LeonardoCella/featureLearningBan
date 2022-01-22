from featLearnBan.policies.SA_FeatLearnBan import SA_FeatLearnBan
from featLearnBan.policies.Random import Random
from featLearnBan.policies.Oful import Oful
from featLearnBan.policies.Lasso import Lasso
from featLearnBan.environment import MTL_ICML as MTL
from featLearnBan.Evaluation import Evaluation

from numpy import arange, cumsum, log, zeros
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import rc


# ===
# RUNNING PARAMETERS
# ===

VERBOSE = True

parser = OptionParser("Usage: %prog [options]", version="%prog 1.0")
parser.add_option('-d', dest = 'd', default = "20", type = 'int', help = "Number of features")
parser.add_option('-r', dest = 'r', default = "5", type = 'int', help = "Rank value")
parser.add_option('-K', dest = 'K', default = "10", type = 'int', help = "Number of arms")
parser.add_option('-T', dest = 'T', default = "20", type = 'int', help = "Number of rounds")
parser.add_option('-N', dest = 'N', default = "10", type = 'int', help = "Number of tasks")
parser.add_option('--nrep', dest = 'nrep', default = "1", type = 'int', help = "Number Repetitions")


(opts, args) = parser.parse_args()
K = opts.K
T = opts.T
N_TASK = opts.N
d = opts.d
r = opts.r
nrep = opts.nrep

noisy_rewards = 1
variance = 1.

assert K > 0, "Not consistent arms number parameter"
assert nrep > 0, "Not consistent number of repetitions"
assert d >= r, "Not consistent sparsity-features relation"
assert T >= r, "Not consistent sparsity-features relation"
assert T > 0, "Not consistent horizon parameter"
assert N_TASK > 0, "Not consistent number of tasks"

if VERBOSE:
    print("RUN Horizon {}, Tasks {}, Arms {}, dimensions {}, rank {}".format(T, N_TASK, K, d, r))

# ===
# INITIALIZATION
# ===
# Policies
policies = {}
policies_name = []

reg = [.1, 1, 10, 100]

# Random Policy
policies["Trace-Norm Bandit"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[0])] 
policies_name.append("Trace-Norm Bandit")
policies["OFUL"] = [Oful(T, d, K, reg[3]) for _ in range(N_TASK)]
policies_name.append("OFUL")
policies["LASSO"] = [Lasso(T, d, K, reg[0]) for _ in range(N_TASK)]
policies_name.append("LASSO")


assert len(policies_name) == len(policies), "Inconsistent Policies"
assert N_TASK >= nrep, "This is required to have a consistent seed"

# Plot parameters
colors = ['r', 'g', 'c', 'b', 'y', 'm', '#ae34eb', '#eb348f', '#000000']
COLORS = {p_name: colors[i] for i, p_name in enumerate(policies_name)}
markers = ["o", "^", "v", "<", ">", ",", "h", "x", "+"]
MARKERS = {p_name: markers[i] for i, p_name in enumerate(policies_name)}
linestyle = ["-", "-", "--", "-.", ":", "--", "-.", ":", "-"]
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
    mtl = MTL(N_TASK, T, K, d, r, variance, p, p_name, noisy_rewards)
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
    ax.set_title('MTL with {} tasks'.format(N_TASK))
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.legend(loc=2)
    plt.savefig(
        'output/ICML_d{}_r{}_T{}_rep{}_N{}_K{}.png'.format(d, r, T, nrep, N_TASK, K))
