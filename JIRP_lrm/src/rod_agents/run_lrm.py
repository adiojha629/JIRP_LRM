import random, time
from reward_machines.reward_machine import RewardMachine
from rod_agents.dqn import DQN
from rod_agents.qrm import QRM
from rod_agents.learning_parameters import LearningParameters
from rod_agents.learning_utils import save_results
from worlds.game import Game
import numpy as np
from data_processing import plot_performance,plot_this
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
    env = Game(env_params)
    rm = RewardMachine(lp.rm_u_max, lp.rm_preprocess, lp.rm_tabu_size, lp.rm_workers, lp.rm_lr_steps, env.get_perfect_rm(), lp.use_perfect_rm)
    actions = env.get_actions()
    policy = None
    train_rewards = []
    rm_scores     = []
    reward_total = 0
    reward_list = [reward_total]
    last_reward  = 0
    step = 0
    # Collecting random traces for learning the reward machine
    print("Collecting random traces...")
    while step < lp.rm_init_steps:
        # running an episode using a random policy
        env.restart()
        trace = [(env.get_events(),0.0)]
        for _ in range(lp.episode_horizon):
            # executing a random action
            a = random.choice(actions)
            #print(env)
            reward, done = env.execute_action(a)
            o2_events = env.get_events()
            reward_total += reward
            reward_list.append(reward)
            trace.append((o2_events,reward))
            step += 1
            # Testing
            if step % lp.test_freq == 0:
                print("Step: %d\tTrain: %0.1f"%(step, reward_total - last_reward))
                train_rewards.append((step, reward_total - last_reward))
                last_reward = reward_total
            # checking if the episode finishes
            if done or lp.rm_init_steps <= step:
                if done: rm.add_terminal_observations(o2_events)
                break 
        # adding this trace to the set of traces that we use to learn the rm
        rm.add_trace(trace)

    # Learning the reward machine using the collected traces
    print("Learning a reward machines...")
    _, info = rm.learn_the_reward_machine()
    rm_scores.append((step,) + info)

    # Start learning a policy for the current rm
    finish_learning = False
    while step < lp.train_steps and not finish_learning: #####
        env.restart()        
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

            # updating the number of steps and total reward
            trace.append((o2_events,reward))
            reward_total += reward
            reward_list.append(reward)
            step += 1

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

def run_lrm_experiments(env_params, lp, rl, n_seed, save,trails):
    time_init = time.time()
    random.seed(n_seed)
    for trail in [0,1]:
        print("Trail: " + str(trail))
        rewards, scores, rm_info,reward_list = run_lrm(env_params, lp, rl)
        if save:
            # Saving the results
            out_folder = "LRM/" + rl + "/active_example/trail_"+str(trail)
            save_results(rewards, scores, rm_info, out_folder, 'lrm', rl, n_seed,reward_list)

    # Showing results
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins\n")
    #print_results()

def print_results():
    plot_dict ={}
    #for loop here
    for trial in range(10):
        file_name = "../results/LRM/lrm-qrm/trail_"+str(trial)+"/officeworld/lrm-lrm-qrm-0_rewards_over_time.txt"
        file = open(file_name)
        lines = file.readlines()
        file.close()
        lines = lines[1:]
        lines = [line.replace("\n","").replace("\t","|") for line in lines]
        reward_list = [int(line[line.find("|")+1:]) for line in lines] #get just the rewards
        #print(reward_list[-1])
        print("rewards for trail "+str(trial)+" obtained")
        if 2 in reward_list: #get rewards per step
            print("editing file")
            list_new = []
            last_num = 0
            for num in reward_list:
                list_new.append(num-last_num)
                last_num = num
            reward_list = list_new.copy()
        #at this point reward list has rewards per step
        print("updateing plot_dict")
        for step in range(int(2e6)):
            reward_at_step = reward_list[step]
            if step in plot_dict.keys():
                plot_dict[step].append(reward_at_step)
            else:
                plot_dict[step] = [reward_at_step]
        #now plot_dict is updated
        print("plot_dict updated for trail #" + str(trial))
    print("calculating percentiles")
    prc_25 = list()
    prc_50 = list()
    prc_75 = list()
    rewards_plot = list()
    steps_plot = list()
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()
    for step in plot_dict.keys():
        if len(current_step) < 10: #if current step has less than 10 elements
            current_25.append(np.percentile(np.array(plot_dict[step]),25))#get the precentiles of values for this step size
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))#append the average value to current step
            #I think that the dictionary holds the values from all 10 trials
        else:#if current step has 10 or more entries, then you remove the last values
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        rewards_plot.append(sum(plot_dict[step])/len(plot_dict[step]))
        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps_plot.append(step)
    #now use these functions to plot the results
    print("now plotting")
    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards (percentiles) vs. Time Step",'LRM-qrm')
    plot_this(steps_plot,rewards_plot,"Average Reward vs. Time Step",'LRM-qrm')
