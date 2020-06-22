import tensorflow as tf

class RL:
    """
    This baseline solves the problem using standard q-learning over the cross product 
    between the RM and the MDP
    """
    def __init__(self):
        self.sess = tf.Session()

    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_opt_restart(self):
        pass

    def test_opt_update(self, o2_events):
        pass

    def test_get_best_action(self, s1, u1):
        return self.get_best_action(s1, u1, 0)

    def learn_if_needed(self):
        raise NotImplementedError("To be implemented")

    def add_experience(self, o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, done):
        raise NotImplementedError("To be implemented")

    def get_best_action(self, s1, u1, epsilon):
        raise NotImplementedError("To be implemented")

        

