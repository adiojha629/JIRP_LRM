import math

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, o1, a, o2):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")

    def compare_to(self, other):
        raise NotImplementedError("To be implemented")

class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def compare_to(self, other):
        return self.get_type() == other.get_type() and self.c == other.c

    def get_reward(self,o1, a, o2):
        return self.c

class EmpiricalRewardFunction(RewardFunction):
    """
    Defines a reward that is empirically estimated and it also depends on the observations of the current state
    """
    def __init__(self):
        super().__init__()
        self.reward_sum   = {}
        self.reward_count = {}

    def get_type(self):
        return "empirical"

    def compare_to(self, other):
        return False

    def get_reward(self, o1, a, o2):
        if o2 in self.reward_sum:
            return float(self.reward_sum[o2])/float(self.reward_count[o2])
        return 0

    def add_observed_reward(self, o2, r):
        if o2 not in self.reward_sum:
            self.reward_sum[o2]   = 0
            self.reward_count[o2] = 0
        self.reward_sum[o2]   += r
        self.reward_count[o2] += 1

    def get_info(self):
        info = []
        for o2 in self.reward_sum:
            r = self.get_reward(None, None, o2)
            if r != 0:
                info.append("\t%s -> %f"%(o2, r))
        return info


    def show(self):
        for o2 in self.reward_sum:
            r = self.get_reward(None, None, o2)
            if r != 0:
                print("\t%s -> %f"%(o2, r))


class EventBasedRewardFunction(RewardFunction):
    """
    Defines a reward that depends on the detected events that were detected on the current state
    """
    def __init__(self, event2reward):
        super().__init__()
        self.event2reward = event2reward

    def get_type(self):
        return "event_based"

    def compare_to(self, other):
        if self.get_type() != other.get_type() or len(self.event2reward) != len(other.event2reward):
            return False

        for e in self.event2reward:
            if e not in other.event2reward or self.event2reward[e] != other.event2reward[e]:
                return False

        return True

    def get_reward(self, o1, a, o2):
        if o1 in self.event2reward:
            return self.event2reward[o1]
        return 0
