import numpy as np

class FeatureProxy:
    def __init__(self, num_features, num_states):
        self.num_features = num_features
        self.num_states   = num_states

    def get_num_features(self):
        return self.num_states + self.num_features

    def add_state_features(self, s, u_i):
        return np.concatenate((s,self._get_one_hot_vector(u_i))) # adding the DFA state to the features

    def _get_one_hot_vector(self, u_i):
        one_hot = np.zeros((self.num_states), dtype=np.float64)
        one_hot[u_i] = 1.0
        return one_hot