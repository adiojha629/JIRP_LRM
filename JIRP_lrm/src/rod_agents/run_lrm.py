import random, time
from reward_machines.reward_machine import RewardMachine
from rod_agents.dqn import DQN
from rod_agents.qrm import QRM
from rod_agents.learning_parameters import LearningParameters
from rod_agents.learning_utils import save_results
from worlds.game import Game

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
    return train_rewards, rm_scores, rm.get_info()

def run_lrm_experiments(env_params, lp, rl, n_seed, save):
    
    time_init = time.time()
    random.seed(n_seed)
    rewards, scores, rm_info = run_lrm(env_params, lp, rl)
    if save:
        # Saving the results
        out_folder = "LRM/" + rl + "/" + env_params.game_type
        save_results(rewards, scores, rm_info, out_folder, 'lrm', rl, n_seed)

    # Showing results
    print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins\n")
