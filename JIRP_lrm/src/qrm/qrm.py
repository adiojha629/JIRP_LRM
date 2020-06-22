import tensorflow as tf
import os.path, time, random
import numpy as np
from agents.rl import RL
from agents.qrm_buffer import ReplayBuffer, PrioritizedReplayBuffer
from agents.dqn_network import create_net, create_target_updates
from common.schedules import LinearSchedule

"""
Interface:
    - close(self): 
        close the session
    - learn_if_needed(self): 
        learns and update the target networks (if needed)
    - add_experience(self, o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, done):
        Adds the current experience to the experience replay buffer
    - get_best_action(self, s1, u1):
        Returns the best action given the current observation "s1" and RM state "u1"
"""

class QRM(RL):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing the current reward machine
    """
    def __init__(self, lp, num_features, num_actions, reward_machine):
        super().__init__()
        
        # learning parameters
        self.lp = lp 
        self.rm = reward_machine
        self.num_features = num_features
        self.num_actions  = num_actions

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

    def _get_step(self):
        return self.step

    def _add_step(self):
        self.step += 1

    def _create_network(self):
        n_features = self.num_features
        n_actions  = self.num_actions
        n_policies = len(self.rm.get_states())

        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, n_features])
        self.a = tf.placeholder(tf.int32) 
        self.s2 = tf.placeholder(tf.float64, [None, n_features])
        self.done = tf.placeholder(tf.float64, [None, n_policies])
        self.ignore = tf.placeholder(tf.float64, [None, n_policies])
        self.rewards = tf.placeholder(tf.float64, [None, n_policies])
        self.next_policies = tf.placeholder(tf.int32, [None, n_policies])
        self.IS_weights = tf.placeholder(tf.float64) # Importance sampling weights for prioritized ER

        # Adding one policy per state in the RM
        self.policies = []
        for i in range(n_policies):
            # adding a policy of the RM state "i"
            policy = PolicyDQN("qrm_%d"%i, self.lp, n_features, n_actions, self.sess, self.s1, self.a, self.s2, self.IS_weights)
            self.policies.append(policy)

        # connecting all the networks into one big net
        self._reconnect()        

    def _reconnect(self):
        # Redefining connections between the different DQN networks
        n_policies = len(self.policies)
        batch_size = self.lp.batch_size
        
        # concatenating q_target of every policy
        Q_target_all = tf.concat([self.policies[i].get_q_target_value() for i in range(len(self.policies))], 1)

        # Indexing the right target next policy
        aux_range = tf.reshape(tf.range(batch_size),[-1,1])
        aux_ones = tf.ones([1, n_policies], tf.int32)
        delta = tf.matmul(aux_range * n_policies, aux_ones) 
        Q_target_index = tf.reshape(self.next_policies+delta, [-1])
        Q_target_flat = tf.reshape(Q_target_all, [-1])
        Q_target = tf.reshape(tf.gather(Q_target_flat, Q_target_index),[-1,n_policies]) 
        # Obs: Q_target is batch_size x n_policies tensor such that 
        #      Q_target[i,j] is the target Q-value for policy "j" in instance 'i'

        # Matching the loss to the right Q_target
        for i in range(n_policies):
            p = self.policies[i]
            # Adding the critic trainer
            p.add_optimizer(self.rewards[:,i], self.done[:,i], Q_target[:,i], self.ignore[:,i])
            # Now that everything is set up, we initialize the weights
            p.initialize_variables()
        
        # Auxiliary variables to train all the critics, actors, and target networks
        self.train = []
        for i in range(n_policies):
            p = self.policies[i]
            if self.lp.prioritized_replay:
                self.train.append(p.td_error)
            self.train.append(p.train)

    def get_best_action(self, s1, u1, epsilon):
        if self._get_step() <= self.lp.learning_starts or random.random() < epsilon:
            # epsilon greedy
            return random.randrange(self.num_actions)
        policy = self.policies[u1]
        s1 = s1.reshape((1,self.num_features))
        return self.sess.run(policy.get_best_action(), {self.s1: s1})[0]

    def _train(self, s1, a, s2, rewards, next_policies, done, ignore, IS_weights):
        # Learning
        values = {self.s1: s1, self.a: a, self.s2: s2, self.rewards: rewards, self.next_policies: next_policies, 
                  self.done: done, self.ignore: ignore, self.IS_weights: IS_weights}
        res = self.sess.run(self.train, values)
        if self.lp.prioritized_replay:
            # Computing new priorities (max of the absolute td-errors)
            td_errors = np.array([np.abs(td_error) for td_error in res if td_error is not None])
            td_errors_max = np.max(td_errors, axis=0) 
            return td_errors_max

    def _learn(self):
        if self.lp.prioritized_replay:
            experience = self.replay_buffer.sample(self.lp.batch_size, beta=self.beta_schedule.value(self._get_step()))
            s1, a, s2, rewards, next_policies, done, ignore, weights, batch_idxes = experience
        else:
            s1, a, s2, rewards, next_policies, done, ignore = self.replay_buffer.sample(self.lp.batch_size)
            weights, batch_idxes = None, None
        td_errors = self._train(s1, a, s2, rewards, next_policies, done, ignore, weights) # returns the absolute td_error
        if self.lp.prioritized_replay:
            new_priorities = np.abs(td_errors) + self.lp.prioritized_replay_eps
            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

    def _update_target_network(self):
        for i in range(len(self.policies)):
            self.policies[i].update_target_networks()

    def learn_if_needed(self): 
        # Learning
        if self._get_step() > self.lp.learning_starts and self._get_step() % self.lp.train_freq == 0:
            self._learn()

        # Updating the target networks
        if self._get_step() > self.lp.learning_starts and self._get_step() % self.lp.target_network_update_freq == 0:
            self._update_target_network()

    def add_experience(self, o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, done):
        # NOTE:
        #   - The reward estimation might change over time
        #   - However, we are adding a fixed reward to the buffer (for simplicity)
        #   - In the future, we might try to recompute the reward every time the experience is sampled

        # Using the RM to compute the rewards, next policies, and 
        # whether it is a terminal transition or it should be ignored
        n_policies = len(self.policies)
        rewards, next_policies, done, ignore = [], [], [], []
        for ui in range(n_policies):
            ui_r  = self.rm.get_reward(ui, o1_events, a, o2_events)
            ui_np = self.rm.get_next_state(ui, o2_events)
            ui_d  = self.rm.is_terminal_observation(o2_events)
            # NOTE: We ignore transitions that are impossible (as explained in Sect. 5 of the paper)
            ui_ig = self.rm.is_observation_impossible(ui, o1_events, o2_events)

            rewards.append(ui_r)
            next_policies.append(ui_np)
            done.append(float(ui_d))
            ignore.append(float(ui_ig))

        # Adding this experience to the replay buffer
        self.replay_buffer.add(o1_features, a, o2_features, rewards, next_policies, done, ignore)
        self._add_step()


class PolicyDQN:
    def __init__(self, policy_name, lp, n_features, n_actions, sess, s1, a, s2, IS_weights):
        self.sess = sess
        self.scope_name = policy_name
        self.n_features = n_features
        self.n_actions  = n_actions
        self.lp = lp
        self.s1 = s1
        self.a  = a
        self.s2 = s2
        self.IS_weights = IS_weights
        self._create_network()

    def _create_network(self):
        
        with tf.variable_scope(self.scope_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            with tf.variable_scope("q_network") as scope:
                q_values, q_values_weights = create_net(self.s1, self.n_features, self.n_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
                if self.lp.use_double_dqn:
                    scope.reuse_variables()
                    q2_values, _ = create_net(self.s2, self.n_features, self.n_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
            with tf.variable_scope("q_target"):
                q_target, q_target_weights = create_net(self.s2, self.n_features, self.n_actions, self.lp.num_neurons, self.lp.num_hidden_layers)
            update_target = create_target_updates(q_values_weights, q_target_weights)

            # Q_values -> get optimal actions
            best_action = tf.argmax(q_values, 1)

            # getting current value for q(s1,a)
            action_mask = tf.one_hot(indices=self.a, depth=self.n_actions, dtype=tf.float64)
            q_current = tf.reduce_sum(tf.multiply(q_values, action_mask), 1)
            
            # getting the target q-value for the best next action
            if self.lp.use_double_dqn:
                # DDQN
                best_action_mask = tf.one_hot(indices=tf.argmax(q2_values, 1), depth=self.n_actions, dtype=tf.float64)
                q_target_value = tf.reshape(tf.reduce_sum(tf.multiply(q_target, best_action_mask), 1), [-1,1])
            else:
                # DQN
                q_target_value = tf.reshape(tf.reduce_max(q_target, axis=1), [-1,1])
            
            # It is important to stop the gradients so the target network is not updated by minimizing the td-error
            q_target_value = tf.stop_gradient(q_target_value)

        # Adding relevant networks to the state properties
        self.best_action = best_action
        self.q_current = q_current
        self.q_target_value = q_target_value
        self.update_target = update_target
                    
    def add_optimizer(self, reward, done, q_target, ignore):
        with tf.variable_scope(self.scope_name): # helps to give different names to this variables for this network
            # computing td-error 'r + gamma * max Q_t'
            q_max    = q_target * (1.0 - done)
            td_error = self.q_current - (reward + self.lp.gamma * q_max)
            self.td_error = td_error * (1.0 - ignore) # setting to zero all the experiences that should be ignored

            # setting loss function
            if self.lp.prioritized_replay: 
                # prioritized experience replay
                huber_loss = 0.5 * tf.square(self.td_error) # without clipping
                loss = tf.reduce_mean(self.IS_weights * huber_loss) # weights fix bias in case of using priorities
            else:
                # standard experience replay
                loss = 0.5 * tf.reduce_sum(tf.square(self.td_error)) 
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lp.lr)
            self.train = optimizer.minimize(loss=loss)

    def initialize_variables(self):
        # Initializing the network values
        self.sess.run(tf.variables_initializer(self._get_network_variables()))
        self.update_target_networks() #copying weights to target net

    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)

    def update_target_networks(self):
        self.sess.run(self.update_target)

    def get_best_action(self):
        return self.best_action

    def get_q_target_value(self):
        return self.q_target_value

