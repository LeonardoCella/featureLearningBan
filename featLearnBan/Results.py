'Utility class that manages the results of MAB experiments'
from numpy import int32, zeros


class Result:
    """Class that analyzes the outcome of a bandit experiment"""

    def __init__(self, horizon):
        # Initially all the rounds have no choices or rewards.
        self.choices = zeros(horizon, dtype=int32)
        self.rewards = zeros(horizon)

    def store(self, t, choice, rwd):
        self.choices[t] = choice
        self.rewards[t] = rwd

    def getNbArms(self):
        return self.K

    def getRewards(self):
        return self.rewards

    def getChoices(self):
        return self.choices

    def __repr__(self):
        return "<Result choices:%s \n rewards %s>" % (self.choices, self.rewards)
