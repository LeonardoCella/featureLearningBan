__version__ = "0.1"

from pandas import concat, DataFrame, read_pickle, read_table
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from numpy import array, asarray, concatenate, diag, dot, hstack, matrix, mean, newaxis, pad, unique, vectorize, vstack, \
    zeros
from numpy.random import RandomState
from numpy.linalg import svd
from pickle import dump, load
from os import listdir, path
from pathlib import Path

here = path.dirname(__file__)


class Movielens():

    def __init__(self, N, d, K, T):
        print("Movielens INIT N {} d {} K {} T {}".format(N, d, K, T))
        self._N_ground = 100
        self._N = N
        self._d = d
        self._K = K
        self._minT = T
        self._rs = RandomState(1)
        # Preprocessing and Context Generation
        self._shrinkItem = 500
        self._shrinkUser = 200
        self._ratings_preprocessing()
        self._context_mng()  # Create self._arms
        self._users_mng()  # Create self._latentUsers
        self._tasks_mng()  # Create self.Xtasks and self._listUID

    def get_context(self, j, t):
        contexts_jt = self.Xtasks[j][t, :]
        return contexts_jt

    def get_rwd(self, j, t):
        rwds = dot(self.get_context(j, t), self._latentUsers.loc[self._listUID[j], :].values)

        if all(r < 0 for r in rwds):
            rwds = rwds - 1 * min(rwds)

        # Rescale until 0.0 - 1.0
        while max(rwds) < 1.0:
            if max(rwds * 10) > 1.0:
                break
            else:
                rwds *= 10

        while max(rwds) > 1.0:
            if max(rwds / 10) < 0.1:
                break
            else:
                rwds /= 10
        return rwds

    def _tasks_mng(self):
        '''Rating Matrix shrinkage by users and items'''

        list_users = list(self._latentUsers.index)
        considered_users = []
        n_bad = self._minT * (self._K - 1)
        all_itemsID = set(self._urm_df.columns.values)

        task_path = "sparseLinban/arms/MovieLens/XTasks_N{}_d{}_T{}_K{}.pkl".format(self._N, self._d, self._minT,
                                                                                    self._K)
        users_path = "sparseLinban/arms/MovieLens/LUsers_N{}_d{}_T{}_K{}.pkl".format(self._N, self._d, self._minT,
                                                                                     self._K)

        # If ready, load the tasks-listuserID-latentUserRepresentation
        if Path(task_path).exists() and Path(users_path).exists():
            # print("Tasks ready")
            with open(Path(task_path), 'rb') as ContextsHandler:
                self.Xtasks = load(ContextsHandler)
                ContextsHandler.close()
            with open(Path(users_path), 'rb') as ContextsHandler:
                self._listUID = load(ContextsHandler)
                ContextsHandler.close()
        else:
            self._rs.shuffle(list_users)
            print("There are {} many users".format(len(list_users)))
            first = True
            # Single Task Definition
            for i, u in enumerate(list_users):
                rated_df = self._urm_byrecords[self._urm_byrecords['UserID'] == u]
                liked_byu_set = set(rated_df.loc[rated_df['rating'] >= 3].ItemID.ravel())
                notliked_byu = list(set(rated_df.ItemID.ravel()) - liked_byu_set)
                liked_byu = list(liked_byu_set)

                # User Jump condition: not enough good
                if len(liked_byu) < self._minT:
                    continue

                positive = self._rs.choice(liked_byu, self._minT)
                remaining = list(all_itemsID - set(liked_byu))
                positive = array(positive).reshape((self._minT, 1))

                # User Jump condition: not enough bad
                if len(remaining) < n_bad:
                    continue

                considered_users.append(u)

                #OLD
#                negative = array(self._rs.choice(remaining, n_bad))
#                suboptimals = negative.reshape((self._minT, self._K - 1))
                suboptimals = vstack([array(self._rs.choice(remaining, self._K - 1)) for _ in range(self._minT)])

                task_i = concatenate((suboptimals, positive), axis=1)
                # ID - Arms Translation
                for t in range(self._minT):
                    armsID_t = task_i[t, :]
                    contexts_t = asarray([self._arms.loc[arm_id][0] for arm_id in armsID_t])
                    contexts_t = contexts_t[newaxis, :]
                    if t == 0:
                        arms_task_i = contexts_t
                    else:
                        arms_task_i = vstack((arms_task_i, contexts_t))

                if first:
                    self.Xtasks = [arms_task_i]
                    first = False
                else:
                    self.Xtasks.append(arms_task_i)

                # Additional Termination Condition
                if len(considered_users) >= self._N:
                    break

            if len(considered_users) < self._N:
                print("{} tasks are not enough".format(len(considered_users)))
                exit(0)
            self._listUID = considered_users
            asarray(self.Xtasks).dump(task_path)
            asarray(considered_users).dump(users_path)
            return

    def _context_mng(self):
        '''Rating Matrix shrinkage by users and items'''

        # If ready, load the contexts
        path = "sparseLinban/arms/MovieLens/contexts_SVD{}.pkl".format(self._d)
        if Path(path).exists():
            # print("Contexts ready")
            with open(path, 'rb') as ContextsHandler:
                self._arms = load(ContextsHandler)
                ContextsHandler.close()
        else:
            # print("Contexts generation")
            self._contexts_generation('userID')
            self._arms.to_pickle(path)

    def _contexts_generation(self, key):
        '''URM decomposition by truncatedSVD'''

        X_np = self._urm_df.loc[:, :].T.values
        U, s, Vh = svd(X_np, full_matrices=True)
        S = diag(s)
        self._arms = DataFrame(dot(U[:, :self._d], S[:self._d, :self._d]), index=self._urm_df.columns)
        self._arms["context"] = self._arms.values.tolist()
        self._arms = self._arms.drop([i for i in range(self._d)], axis=1)
        return

    def _users_mng(self):
        '''Latent Users Creation'''
        path = 'sparseLinban/arms/MovieLens/usersDF_SVD{}.pkl'.format(self._d)
        if Path(path).exists():
            self._latentUsers = read_pickle(path)
        else:
            X_np = self._urm_df.loc[:, :].values
            U, _, _ = svd(X_np, full_matrices=True)
            X_std = U[:, :self._d]  # Latent Representation
            full_users = DataFrame(U, index=self._urm_df.index)
            self._latentUsers = DataFrame(X_std, index=self._urm_df.index)  # Lantent DF
            self._latentUsers.to_pickle(path)

    def _ratings_preprocessing(self):
        '''Matrix shrinking'''

        with open("sparseLinban/arms/MovieLens/ratings.dat", 'r') as f:
            urm_byrecords = read_table(f, sep="::", header=None, names=['UserID', 'ItemID', 'rating', 'timestamp'],
                                       engine='python')
        urm_byrecords = urm_byrecords.drop(['timestamp'], axis=1)
        # Filter Low Rated Movies
        urm_byrecords = urm_byrecords[urm_byrecords.groupby('ItemID')['ItemID'].transform('count').ge(self._shrinkItem)]
        # Filter Low Rater Users
        urm_byrecords = urm_byrecords[urm_byrecords.groupby('UserID')['UserID'].transform('count').ge(self._shrinkUser)]
        # Remove rows with NaN
        urm_byrecords.dropna(axis=0, how='any', inplace=True)
        # Convert from Records to DF
        urm_df = urm_byrecords.pivot_table(index="UserID", columns="ItemID", values="rating", fill_value=0)
        # Storing
        self._urm_byrecords = urm_byrecords
        self._urm_df = urm_df
        print("URM shape {}".format(self._urm_df.shape))
        return

    def hasEnoughRounds(self, j):
        return self._Xtasks[j].shape[0] >= self._minT

    def get_size(self):
        return self._d
