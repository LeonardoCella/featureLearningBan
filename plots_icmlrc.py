from featLearnBan.policies.SA_FeatLearnBan import SA_FeatLearnBan
from featLearnBan.policies.Random import Random
from featLearnBan.policies.Oful import Oful
from featLearnBan.policies.Lasso import Lasso
from featLearnBan.environment import MTL_ICMLrc as MTL
from featLearnBan.Evaluation import Evaluation

from numpy import arange, cumsum, log, zeros
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import rc
import json

# ===
# RUNNING PARAMETERS
# ===

VERBOSE = True

parser = OptionParser("Usage: %prog [options]", version="%prog 1.0")
parser.add_option('-T', dest = 'T', default = "10", type = 'int', help = "Number of Rounds")
parser.add_option('--nrep', dest = 'nrep', default = "1", type = 'int', help = "Number Repetitions")

(opts, args) = parser.parse_args()

T_in = opts.T
nrep = opts.nrep

noisy_rewards = True
variance = 1.

# ===
# CONFIGURATIONS ITERATION
# ===
list_d = [20]
list_N = [10, 20, 50, 70]
list_T = [T_in]
list_r = [10]
list_K = [10]


for T in list_T: 
    for r in list_r:
        for K in list_K:
            # Results Containers
            dict_avg_results = {}
            dict_std_results = {}

            # Plots are done at this level
            for d in list_d:
                for N_TASK in list_N:
                    # CONSISTENCY CHECK
                    assert K > 1, "Not relevant arms number parameter"
                    assert nrep > 0, "Not consistent number of repetitions"
                    assert d >= r, "Not consistent features-rank relation"
                    assert N_TASK >= r, "Not consistent tasks-rank relation"
                    assert T > 0, "Not consistent horizon parameter"
                    assert r > 0, "Not consistent rank"

                    if VERBOSE:
                        print("RUN Horizon {}, Tasks {}, Arms {}, dimensions {}, rank {}".format(T, N_TASK, K, d, r))

                    # ===
                    # INITIALIZATION
                    # ===
                    # Policies
                    policies = {}
                    policies_name = []

                    reg = [.1, 1, 10, 100]
                    
                    policies["Random"] = [Random(K) for _ in range(N_TASK)]
                    policies_name.append("Random")
                    policies["Trace-Norm Bandit"] = [SA_FeatLearnBan(N_TASK, T, d ,K, reg[0])] 
                    policies_name.append("Trace-Norm Bandit")
                    policies["OFUL"] = [Oful(T, d, K, reg[2]) for _ in range(N_TASK)]
                    policies_name.append("OFUL")


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

                        # KEY DEFINITION AND RESULT CONTAINER PREPARATION
                        key = p_name + "_" + str(d) + "_" + str(N_TASK) 
                        dict_avg_results.setdefault(key, 0.0)
                        dict_std_results.setdefault(key, 0.0)
                        
                        print("\n=======NEW RUN=======")
                        print("===POLICY {}===".format(p_name))
                        # Here each policy p is a list of N_TASK policies except for SA_FeatLearnBan
                        mtl = MTL(N_TASK, T, K, d, r, variance, p, p_name, noisy_rewards)
                        evaluation = Evaluation(mtl, T, p_name, nrep, N_TASK)
                        results.append(evaluation.getResults())  # Result Class: (Policy Name, Mean Rewards, Std Rewards)

                    if VERBOSE:
                        # ===
                        # RESULTS EXTRACTION
                        # ===

                        # Rewards Extraction
                        POLICY_AVGS = []  # cum sum mean rwd
                        POLICY_STD = []
                        POLICY_N = []
                        round_indexes = arange(T * N_TASK)

                        for name, avg, std in results:  # list of (policy name, meanRwds, stdRwds)
                            cumsumavg = cumsum(avg)
                            # plt.fill_between(round_indexes, cumsumavg - std / 2, cumsumavg + std / 2, alpha=0.5, color=COLORS[name])
                            # plt.plot(round_indexes, cumsumavg, color=COLORS[name], marker=MARKERS[name],
                            #         label="Policy {}".format(name))
                            # print("\nPOL {}, RWDS EVERY 10 rounds: {}".format(name, [cumsumavg[i] for i in arange(0,len(cumsumavg),10)]))
                            print("POL {} FINAL {}".format(name, cumsumavg[-1]))
                            key = name + "_" + str(d) + "_" + str(N_TASK) 
                            dict_avg_results[key] = cumsumavg[-1]
                            dict_std_results[key] = std[-1]

            # ITERATED OVER FEATURES AND TASKS. FIXED r,K,T
            # print(dict_avg_results.items())
            # print(dict_std_results.items())


            # STORAGE
            with open("data/avgRC_d{}_r{}_T{}_K{}_nrep{}.json".format(d, r, T, K, nrep), 'w') as favg:
                json.dump(dict_avg_results, favg)
            with open("data/stdRC_d{}_r{}_T{}_K{}_nrep{}.json".format(d, r, T, K, nrep), 'w') as fstd:
                json.dump(dict_std_results, fstd)

            # Result extraction by policies fixed d
            for d in list_d: 
                fig = plt.figure(1)
                ax = fig.add_subplot(1, 1, 1)
                for p_name in policies_name:
                    avg_byTask = []
                    std_byTask = []
                    for N_TASK in list_N:
                        key = p_name + "_" + str(d) + "_" + str(N_TASK)
                        avg_byTask.append(dict_avg_results[key])
                        std_byTask.append(dict_std_results[key])

                    plt.errorbar(list_N, avg_byTask, std_byTask, label = p_name + str(d) + "_" + str(K) , color = COLORS[p_name], marker = MARKERS[p_name])
                ax.set_xlabel('Tasks')
                ax.set_ylabel('Cumulative Rewards')
                ax.set_title('MTL by Tasks')
                ax.yaxis.grid(True)

                # Save the figure and show
                plt.legend(loc=2)
                plt.savefig(
                    'output/ICML_rndcov_byTASKS_d{}_r{}_T{}_K{}_nrep{}.png'.format(d, r, T, K, nrep))
                
                # Clean current axes
                ax.cla()
