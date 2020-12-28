import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
from rod_agents.rl import RL
from rod_agents.dqn_network import create_net, create_target_updates
from rod_agents.dqn_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rod_agents.feature_proxy import FeatureProxy
from common.schedules import LinearSchedule
import random

class DQN(RL):
    """
    This baseline solves the problem using standard q-learning over the cross product 
    between the RM and the MDP
    """
    def __init__(self, lp, num_features, num_actions, reward_machine):
        super().__init__()
        # learning parameters
        self.lp = lp 
        self.policy_name = 'dqn_network'

        # This proxy adds the machine state representation to the MDP state
        num_states = 1 if reward_machine is None else len(reward_machine.get_states())
        self.feature_proxy = FeatureProxy(num_features, num_states)
        self.num_actions  = num_actions
        self.num_features = self.feature_proxy.get_num_features()

        # Creating the network
        self.sess = tf.Session()
        self._create_network()

        # create experience replay buffer
        if self.lp.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(lp.buffer_size, alpha=lp.prioritized_replay_alpha)
            if lp.prioritized_replay_beta_iters is None:
                lp.prioritized_replay_beta_iters = lp.train_steps
            self.beta_schedule = LinearSchedule(lp.prioritized_replay_beta_iters, initial_p=lp.prioritized_replay_beta0, final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(lp.buffer_size)
            self.beta_schedule = None

        # count of the number of environmental steps
        self.step = 0

    def _create_network(self):
        total_features = self.num_features
        total_actions = self.num_actions

        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, total_features])
        self.a = tf.placeholder(tf.int32)
        self.r = tf.placeholder(tf.float64)
        self.s2 = tf.placeholder(tf.float64, [None, total_features])
        self.done = tf.placeholder(tf.float64)
        self.IS_weights = tf.placeholder(tf.float64) # Importance sampling weights for prioritized ER

        # Creating target and current networks
        with tf.variable_scope(self.policy_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            with tf.variable_scope("q_network") as scope:
                q_values, q_values_weights = create_net(self.s1, total_features, total_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
                if self.lp.use_double_dqn:
                    scope.reuse_variables()
                    q2_values, _ = create_net(self.s2, total_features, total_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
            with tf.variable_scope("q_target"):
                q_target, q_target_weights = create_net(self.s2, total_features, total_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
            self.update_target = create_target_updates(q_values_weights, q_target_weights)

            # Q_values -> get optimal actions
            self.best_action = tf.argmax(q_values, 1)

            # Optimizing with respect to q_target
            action_mask = tf.one_hot(indices=self.a, depth=total_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)

            if self.lp.use_double_dqn:
                # DDQN
                best_action_mask = tf.one_hot(indices=tf.argmax(q2_values, 1), depth=total_actions, dtype=tf.float64)
                q_max = tf.reduce_sum(tf.multiply(q_target, best_action_mask), 1)
            else:
                # DQN
                q_max = tf.reduce_max(q_target, axis=1)

            # Computing td-error and loss function
            q_max = q_max * (1.0-self.done) # dead ends must have q_max equal to zero
            q_target_value = self.r + self.lp.gamma * q_max
            q_target_value = tf.stop_gradient(q_target_value)
            if self.lp.prioritized_replay: 
                # prioritized experience replay
                self.td_error = q_current - q_target_value
                huber_loss = 0.5 * tf.square(self.td_error) # without clipping
                loss = tf.reduce_mean(self.IS_weights * huber_loss) # weights fix bias in case of using priorities
            else:
                # standard experience replay
                loss = 0.5 * tf.reduce_sum(tf.square(q_current - q_target_value))
            
            # Defining the optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lp.lr)
            self.train = optimizer.minimize(loss=loss)
            
        # Initializing the network values
        self.sess.run(tf.variables_initializer(self._get_network_variables()))
        self._update_target_network() #copying weights to target net

    def _train(self, s1, a, r, s2, done, IS_weights):
        if self.lp.prioritized_replay: 
            _, td_errors = self.sess.run([self.train,self.td_error], {self.s1: s1, self.a: a, self.r: r, self.s2: s2, self.done: done, self.IS_weights: IS_weights})
        else:
            self.sess.run(self.train, {self.s1: s1, self.a: a, self.r: r, self.s2: s2, self.done: done})
            td_errors = None
        return td_errors

    def _get_step(self):
        return self.step

    def _add_step(self):
        self.step += 1

    def _learn(self):
        if self.lp.prioritized_replay:
            experience = self.replay_buffer.sample(self.lp.batch_size, beta=self.beta_schedule.value(self._get_step()))
            s1, a, r, s2, done, weights, batch_idxes = experience
        else:
            s1, a, r, s2, done = self.replay_buffer.sample(self.lp.batch_size)
            weights, batch_idxes = None, None

        td_errors = self._train(s1, a, r, s2, done, weights) # returns the absolute td_error
        if self.lp.prioritized_replay:
            new_priorities = np.abs(td_errors) + self.lp.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def _update_target_network(self):
        self.sess.run(self.update_target)

    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy_name)

    def learn_if_needed(self):
        # Learning
        if self._get_step() > self.lp.learning_starts and self._get_step() % self.lp.train_freq == 0:
            self._learn()

        # Updating the target networks
        if self._get_step() > self.lp.learning_starts and self._get_step() % self.lp.target_network_update_freq == 0:
            self._update_target_network()

    def add_experience(self, o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, done):
        s1 = self.feature_proxy.add_state_features(o1_features, u1)
        s2 = self.feature_proxy.add_state_features(o2_features, u2)
        self.replay_buffer.add(s1, a, reward, s2, done)
        self._add_step()

    def get_best_action(self, s1, u1, epsilon):
        if self._get_step() <= self.lp.learning_starts or random.random() < epsilon:
            # epsilon greedy
            return random.randrange(self.num_actions)
        s1 = self.feature_proxy.add_state_features(s1, u1).reshape((1,self.num_features))
        return self.sess.run(self.best_action, {self.s1: s1})[0]
