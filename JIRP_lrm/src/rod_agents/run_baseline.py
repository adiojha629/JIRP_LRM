import random, time
import numpy as np
from reward_machines.reward_machine import RewardMachine
from agents.dqn import DQN
from agents.learning_parameters import LearningParameters
from agents.learning_utils import save_results
from worlds.game import Game

"""
- This code runs standard DQN (it doesn't learn the reward machine)
- It also evaluates the performance of optimal policies
"""

def run_baseline(env_params, lp, rl, k_order):
    """
    This code learns a reward machine from experience and uses dqn to learn an optimal policy for that RM:
        - 'env_params' is the environment parameters
        - 'lp' is the set of learning parameters
    Returns the training rewards
    """
    # Initializing parameters and the game
    env = Game(env_params)
    actions = env.get_actions()
    policy = None
    train_rewards = []
    reward_total = 0
    last_reward  = 0
    step = 0

    # Start learning a policy for the current rm
    while step < lp.train_steps:
        env.restart()
        o1_events   = env.get_events()
        o1_features = env.get_features()
        # computing the stack of features for o1
        k_prev_obs = [np.zeros(len(o1_features)) for _ in range(k_order-1)] # saves the k-previous observations
        k_prev_obs.insert(0, o1_features)
        o1_stack = np.concatenate(tuple(k_prev_obs), axis=None)
        for _ in range(lp.episode_horizon):

            # reinitializing the policy if the rm changed
            if policy is None:
                if rl == "dqn":
                    policy = DQN(lp, k_order * len(o1_features), len(actions), None)
                elif rl == "human":
                    policy = None
                else:
                    assert False, "RL approach is not supported yet"            

            # selecting an action using epsilon greedy
            if rl == "human":
                if random.random() < 0.1: a = random.randrange(4)
                else: a = env.get_optimal_action().value
            else:
                a = policy.get_best_action(o1_stack, 0, lp.epsilon)

            # executing a random action
            reward, done = env.execute_action(a)
            o2_events   = env.get_events()
            o2_features = env.get_features()

            # Appending the new observation and computing the stack of features for o2
            k_prev_obs.insert(0, o2_features)
            k_prev_obs.pop()
            o2_stack = np.concatenate(tuple(k_prev_obs), axis=None)

            # updating the number of steps and total reward
            reward_total += reward
            step += 1

            if rl != "human":
                # Saving this transition
                policy.add_experience(o1_events, o1_stack, 0, a, reward, o2_events, o2_stack, 0, float(done))

                # Learning and updating the target networks (if needed)
                policy.learn_if_needed()

            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total

            # checking if the episode finishes
            if done or lp.train_steps <= step: 
                break 

            # Moving to the next state
            o1_events, o1_features, o1_stack = o2_events, o2_features, o2_stack

    # closing the policy
    if policy is not None:
        policy.close()
        policy = None

    # return the trainig rewards
    return train_rewards


def run_baseline_experiments(env_params, lp, rl, k_order, n_seed, save):
    
    time_init = time.time()
    random.seed(n_seed)
    results = run_baseline(env_params, lp, rl, k_order)
    if save:
        # Saving the results
        out_folder = "BASELINE/" + rl + "/" + env_params.game_type
        rl_alg = rl if k_order == 1 else "%d-order_%s"%(k_order, rl)
        save_results(results, None, None, out_folder, 'baseline', rl_alg, n_seed)

    # Showing results
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")
