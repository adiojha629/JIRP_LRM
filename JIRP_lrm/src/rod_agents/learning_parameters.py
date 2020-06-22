class LearningParameters:
    def __init__(self):
        # default values
        self.prioritized_replay = False
        self.use_perfect_rm = False

    def set_test_parameters(self, test_freq):
        self.test_freq  = test_freq

    def set_rm_learning(self, rm_init_steps, rm_u_max, rm_preprocess, rm_tabu_size, rm_lr_steps, rm_workers):
        self.rm_init_steps = rm_init_steps # number of initial steps to run before learning the RM
        self.rm_u_max      = rm_u_max     # max number of states for the reward machine
        self.rm_preprocess = rm_preprocess # True for preprocessing the trace
        self.rm_tabu_size  = rm_tabu_size # tabu list size
        self.rm_lr_steps   = rm_lr_steps   # Number of learning steps for Tabu search
        self.rm_workers    = rm_workers # number of threads 

    def set_rl_parameters(self, gamma, train_steps, episode_horizon, epsilon, max_learning_steps):
        self.gamma = gamma
        self.train_steps = train_steps
        self.episode_horizon = episode_horizon
        self.epsilon = epsilon
        self.max_learning_steps = max_learning_steps

    def set_perfect_rm(self):
        # HACK: only for debugging purposes, we can see the performance that can be achieved by a handcrafted perfect RM
        self.use_perfect_rm = True

    def set_prioritized_experience_replay(self, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
                                          prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-6):
        self.prioritized_replay = True
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.prioritized_replay_eps = prioritized_replay_eps

    def set_deep_rl(self, lr, learning_starts, train_freq, target_network_update_freq, 
                    buffer_size, batch_size, use_double_dqn, num_hidden_layers, num_neurons):
        self.tabular_case = False
        self.lr = lr
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_network_update_freq = target_network_update_freq
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        # Network architecture
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
