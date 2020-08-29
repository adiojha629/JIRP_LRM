import random, time
from reward_machines.reward_machine import RewardMachine
from rod_agents.dqn import DQN
from rod_agents.qrm import QRM
from rod_agents.learning_parameters import LearningParameters
from rod_agents.learning_utils import save_results
from worlds.game import Game
import numpy as np
from data_processing import plot_performance,plot_this
import os, pickle
import matplotlib.pyplot as plt
"""
- Pseudo-code:
    - Run 'n' random episodes until completion or timeout
    - Learn an RM using those traces
    - Learn a policy for the learned RM until reaching a contradition
    - Add the contradicting trace and relearn the RM
    - Relearn a policy for the new RM from stratch
NOTE:
    - The previous approach can be improved in several ways, but I like its simplicity
    - The policies might be learned using DQN or QRM
"""

def run_lrm(env_params, lp, rl):
    """
    This code learns a reward machine from experience and uses dqn to learn an optimal policy for that RM:
        - 'env_params' is the environment parameters
        - 'lp' is the set of learning parameters
    Returns the training rewards
    """
    # Initializing parameters and the game
    env = Game(env_params,label="train")
    rm = RewardMachine(lp.rm_u_max, lp.rm_preprocess, lp.rm_tabu_size, lp.rm_workers, lp.rm_lr_steps, env.get_perfect_rm(), lp.use_perfect_rm)
    actions = env.get_actions()
    policy = None
    train_rewards = []
    rm_scores     = []
    reward_total = 0
    reward_list = []
    last_reward  = 0
    step = 0

    #parameters for testing agent throughout training: 8/7/20
    test_frq = lp.test_freq #how often we test agent 8/7/20
    test_epi_length = lp.test_epi_length #how long we test agent 8/7/20
    #plot_dict = dict() #used for plotting rewards over time from tests 8/7/20
    test_step = 0 #used by plot_dict (variable above) 8/7/20
    test_env = Game(env_params,label="test") #environment used for testing 8/7/2020
    #parameters to study while testing
    #num_of_suc = 0 #how many times does the agent complete an episode in testing 8.7.20
    #time_to_suc = [] #track how long it took agent to complete episode 8.7.20
    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < lp.rm_init_steps:
        # running an episode using a random policy
        env.restart("train")
        trace = [(env.get_events(),0.0)]
        for _ in range(lp.episode_horizon):
            # executing a random action
            a = random.choice(actions)
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            trace.append((o2_events,reward))
            reward_total+=reward
            step += 1
            '''Code for testing agent performance'''
            if step % test_frq == 0: #We test the model if a test_frq number of time steps have passed
                #below we reset the environment, reward, and done variables
                test_env.restart("test")
                test_reward = 0
                test_done = False
                for test_trail in range(test_epi_length): #this is the testing loop
                    if not(test_done): #if an episode isn't complete
                        act = random.choice(actions) #choose a random action
                        test_reward,test_done = test_env.execute_action(act) #execute that action
                    else:#if an episode was completed:
                        #num_of_suc += 1 #increment number of successes but agent to complete an episode 8.7.20
                        #time_to_suc.append(test_trail) #record how long it took to complete an episode 8.7.20
                        break #break out of for loop
                test_step += test_frq #increment test_step by the test_frq
                reward_list.append([test_step,test_reward])
            # Testing (LRM testing, not what we need to compare with JIRP)
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
            # checking if the episode finishes
            if done or lp.rm_init_steps <= step:
                if done:rm.add_terminal_observations(o2_events)
                break
        # adding this trace to the set of traces that we use to learn the rm
        rm.add_trace(trace)
    print("Done with random action")
    # Learning the reward machine using the collected traces
    print("Learning a reward machines...")
    #adi = input("continue?")
    _, info = rm.learn_the_reward_machine()
    rm_scores.append((step,) + info)
    # Start learning a policy for the current rm
    finish_learning = False
    while step < lp.train_steps and not finish_learning: #####
        env.restart("train")
        o1_events   = env.get_events()
        o1_features = env.get_features()
        u1 = rm.get_initial_state()
        trace = [(o1_events, 0.0)]
        add_trace = False
        
        for _ in range(lp.episode_horizon):##### 5000

            # reinitializing the policy if the rm changed
            if policy is None:
                print("Learning a policy for the current RM...")
                if rl == "dqn":
                    policy = DQN(lp, len(o1_features), len(actions), rm)
                elif rl == "qrm" or rl == "lrm-qrm":
                    policy = QRM(lp, len(o1_features), len(actions), rm)
                else:
                    assert False, "RL approach is not supported yet"
        
            # selecting an action using epsilon greedy
            a = policy.get_best_action(o1_features, u1, lp.epsilon)

            # executing a random action
            reward, done = env.execute_action(a)
            o2_events   = env.get_events()
            o2_features = env.get_features()
            u2 = rm.get_next_state(u1, o2_events)
            reward_total+=reward
            # updating the number of steps and total reward
            trace.append((o2_events,reward))
            step += 1
            if step % test_frq == 0: #let's test agent
                test_done = False
                test_reward = 0
                test_env.restart("train")
                test_o1_events   = test_env.get_events()
                test_o1_features = test_env.get_features()
                test_u1 = rm.get_initial_state()
                for _ in range(test_epi_length):
                    if not(test_done):
                        act = policy.get_best_action(test_o1_features, test_u1, lp.epsilon)
                        test_reward,test_done = test_env.execute_action(act)
                        test_o2_events   = test_env.get_events()
                        test_o2_features = test_env.get_features()
                        test_u2 = rm.get_next_state(test_u1, test_o2_events)
                        test_o1_events, test_o1_features, test_u1 = test_o2_events, test_o2_features, test_u2
                        #print("Agent took action")
                    else:
                        break
                        print("Agent finsihed early with policy") #for debugging# Aug 5 14;48; the agent does finish early now
                test_step += test_frq
                reward_list.append([test_step,test_reward])

            # updating the current RM if needed
            rm.update_rewards(u1, o2_events, reward)
            if done: rm.add_terminal_observations(o2_events)
            if rm.is_observation_impossible(u1, o1_events, o2_events):
                # if o2 is impossible according to the current RM, 
                # then the RM has a bug and must be relearned
                add_trace = True
            # Saving this transition
            policy.add_experience(o1_events, o1_features, u1, a, reward, o2_events, o2_features, u2, float(done))
            # Learning and updating the target networks (if needed)
            policy.learn_if_needed()

            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
                # finishing the experiment if the max number of learning steps was reached
                if policy._get_step() > lp.max_learning_steps:
                    finish_learning = True

            # checking if the episode finishes or the agent reaches the maximum number of training steps
            if done or lp.train_steps <= step or finish_learning: 
                break 

            # Moving to the next state
            o1_events, o1_features, u1 = o2_events, o2_features, u2

        # If the trace isn't correctly predicted by the reward machine, 
        # we add the trace and relearn the machine
        if add_trace and step < lp.train_steps and not finish_learning:
            print("Relearning the reward machine...")
            rm.add_trace(trace)
            same_rm, info = rm.learn_the_reward_machine()
            rm_scores.append((step,) + info)
            if not same_rm:
                # if the RM changed, we have to relearn all the q-values...
                policy.close()
                policy = None
            else:
                print("the new RM is not better than the current RM!!")
                #input()

    if policy is not None:
        policy.close()
        policy = None
    # return the trainig rewards
    return train_rewards, rm_scores, rm.get_info(),reward_list

def run_lrm_experiments(env_params, lp, rl, n_seed, save,trails,task):
    time_init = time.time()
    #random.seed(n_seed)
    for trail in trails:
        print("Trail: " + str(trail))
        rewards, scores, rm_info,reward_list = run_lrm(env_params, lp, rl)
        if save:
            # Saving the results
            out_folder = "LRM/" + rl + "/task_"+task+"/trail_"+str(trail)
            save_results(rewards, scores, rm_info, out_folder, 'lrm', rl, n_seed,reward_list)
            print("Results saved to ",out_folder)

    # Showing results
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins\n")
    #print_results()


