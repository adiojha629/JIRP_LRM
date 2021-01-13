import random, time, argparse, os.path
from automata_learning_with_policybank import aqrm
from automata_learning.aqrm import run_aqrm_experiments
from automata_learning.qlearning import run_qlearning_experiments
from baselines.run_dqn import run_dqn_experiments
from baselines.run_hrl import run_hrl_experiments
from tester.tester import Tester
from testerHRL.tester import TesterHRL
from tester_policybank.tester import TesterPolicyBank
from tester.tester_params import TestingParameters
from common.curriculum import CurriculumLearner
from rod_agents.learning_parameters import LearningParameters as Rod_LearningParameters
from qrm.learning_params import LearningParameters
from rod_agents.run_lrm import run_lrm_experiments
import numpy as np
#from run_lrm import run_lrm_experiments
from worlds.game import GameParams
from worlds.grid_world import GridWorldParams
def get_params_craft_world(experiment,experiment_known):
    step_unit = 400

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 1
    learning_params.relearn_period = 30
    learning_params.enter_loop = 10
    learning_params.lr = 1  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 32
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 1000

    # Tabular case
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment,rod = False)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 1500 * step_unit
    curriculum.min_steps = 1

    print("Craft World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum
def get_params_traffic_world(experiment):
    step_unit = 100

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.memory_size = 5
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.tabular_case = False
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.relearn_period = 100
    learning_params.enter_loop = 5
    learning_params.memory_size = 200
    learning_params.buffer_size = 1

    learning_params.lr = 1  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 10
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 10

    # These are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = False
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment,rod = False)


    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 20000*step_unit
    curriculum.min_steps = 1

    print("Traffic World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)


    return testing_params, learning_params, tester, curriculum


#NOTE: experiment_known allows the code to automatically set the learning parameters used in the AFRAI-RL paper
def get_params_office_world(experiment,experiment_known):
    adi = input("Reached get_params_office_world")
    f = open(experiment_known)
    lines = [l.rstrip() for l in f]
    f.close()
    task_str = eval(lines[1])[0]
    task = task_str.find(".txt")
    task = task_str[task-2:-4] #get task number
    print(task_str)
    step_unit = total_num_steps = num_random_action = 0 #variables that vary based on task
    if("7" in task): #check if running abac task
        print("Using learning parameters for Task 7: ABAC")
        step_unit = 200
        total_num_steps = int(1e6)
        num_random_action = 2e5
    elif("9" in task):
        print("Using learning parameters for Task 9: BCABCA")
        step_unit = 800
        total_num_steps = int(2e6)
        num_random_action = 2e5
    elif("10" in task_str):
        print("Using learning parameters for Task 10: CBABCA")
        step_unit = 800
        total_num_steps = int(6e6)
        num_random_action = 2e5
    else:
        print("Default Learning params being used: Not parameters used in experiments in research paper")
        step_unit = 200
        total_num_steps = int(2e6)
        num_random_action = 2e5


    #step_unit = 200

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq =  step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 10
    learning_params.relearn_period = 30
    learning_params.enter_loop = 10
    learning_params.lr = 5e-5  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 10

    # Tabular case
    learning_params.tabular_case = False  
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = False
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 5
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = total_num_steps #9.8.20 -Aditya Ojha
    curriculum.min_steps = 1

    #print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def run_experiment(world, alg_name, experiment_known, experiment_learned, num_times, show_print, show_plots, is_SAT,num_trials):
    """The Below code was commented out, as it is a left over from JIRP. Only code that runs LRM is uncommented"""

    if alg_name == "lrm-qrm" or alg_name == "lrm-dqn": #code for running LRM
        rl = alg_name
        #determine environment: only for LRM setup
        if "active" in world:
            env = "officeworld_active"
        elif "craft" in world:
            env = "craftworld"
        elif "office" in world:
            env = "office_world"
        else:
            print(world + " is not supportted at this time")
        n_seed = 0
        n_workers = 16
        run_lrm_agent(rl,env,n_seed,n_workers,num_trials,experiment_known)
    else: #use non-LRM (ie JIRP) learning setup
        tester = None
        tester_l = None #pycharm won't complie w/out this: "variable referenced before assignment" error
        curriculum = None
        if world == 'officeworld' or "office" in world:
            testing_params_k, learning_params_k, tester, curriculum_k = get_params_office_world(experiment_known,experiment_known)
            testing_params, learning_params, tester_l, curriculum = get_params_office_world(experiment_learned,experiment_known)
            ## allows for 2 sets of testers/curricula: one for previously known (ground truth) and one for learned info
        if world == 'craftworld':
            testing_params_k, learning_params_k, tester, curriculum_k = get_params_craft_world(experiment_known,experiment_known)
            testing_params, learning_params, tester_l, curriculum = get_params_office_world(experiment_learned,experiment_known)
        if world == 'trafficworld':
            testing_params_k, learning_params_k, tester, curriculum_k = get_params_traffic_world(experiment_known)
            testing_params, learning_params, tester_l, curriculum = get_params_traffic_world(experiment_learned)

        if alg_name == "ddqn":
            tester = TesterPolicyBank(learning_params, testing_params, experiment_known)
            run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print)

        if alg_name == "hrl":
            if world == 'craftworld':
                testing_params, learning_params, tester, curriculum = get_params_craft_world("../experiments/craft/tests/ground_truth.txt")
            elif world == 'officeworld':
                testing_params, learning_params, tester, curriculum = get_params_office_world("../experiments/office/tests/ground_truth.txt")
            tester = TesterHRL(learning_params, testing_params, experiment_known)
            run_hrl_experiments(alg_name, tester, curriculum, num_times, show_print, use_rm = False)

        #if (alg_name == "jirp") and (world== "trafficworld") and (not is_SAT):
        #    tester = TesterPolicyBank(learning_params, testing_params, experiment_known)
        #    tester_l = TesterPolicyBank(learning_params, testing_params, experiment_learned)
        #    aqrm.run_aqrm_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots, is_SAT)
        if alg_name == "jirp":
            run_aqrm_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots, is_SAT)

        if alg_name == "qlearning":
            run_qlearning_experiments(alg_name, tester, tester_l, curriculum, num_times, show_print, show_plots)

#lp and learning parameters variables point to the same Learning Parameters class
def run_lrm_agent(rl, env, n_seed, n_workers,num_trails,experiment):
    #variables that chagne based on experiment and environment:
    test_freq = None #how often we test agent
    test_epi_length = None #how long can the agent be tested for
    total_train_step = None  #total number of training steps
    num_random_action = None #number of initial steps agent can take with random action. specific to LRM
    task_str = None #variable to help us find task
    task = None # what the task is
    '''All this above variables are defined and used below'''

    print("\n----------------------")
    print("LRM agent:", rl)
    print("----------------------\n")
    #get task information:
    f = open(experiment)
    lines = [l.rstrip() for l in f]
    f.close()
    # setting the test attributes
    if "office" in env:
        task_str = eval(lines[1])[0]
        if("7" in task_str):
            print("Using learning parameters for OFFICE Task 7: ABAC")
            test_freq = 200
            test_epi_length = 200
            total_train_step = int(1e6)
            num_random_action = int(2e5)
        elif("9" in task_str):
            print("Using learning parameters for OFFICE Task 9: BCABCA")
            test_freq = 800
            test_epi_length = 800
            total_train_step = int(2e6)
            num_random_action = int(2e5)
        elif("10" in task_str):
            print("Using learning parameters for OFFICE Task 10: CBABCA")
            test_freq = 800
            test_epi_length = 800
            total_train_step = int(6e6)
            num_random_action = int(2e5)
        else:
            print("Default Learning params being used: Not parameters used in experiments in research paper")
            test_freq = 200
            test_epi_length = 200
            total_train_step = int(2e6)
            num_random_action = int(2e5)
    elif "craft" in env:
        task_str = eval(lines[2])[0]
        if("7" in task_str):
            print("Using learning parameters for CRAFT Task 7: ABAC")
            test_freq = 400
            test_epi_length = 400
            total_train_step = int(4e5)
            num_random_action = int(8e4)
        elif("9" in task_str):
            print("Using learning parameters for CRAFT Task 9: BCABCA")
            test_freq = 600
            test_epi_length = 600
            total_train_step = int(6e5)
            num_random_action = int(12e4)
        elif("6" in task_str):
            print("Using learning parameters for CRAFT Task 6: Sword")
            test_freq = 400
            test_epi_length = 400
            total_train_step = int(4e5)
            num_random_action = int(8e4)
        elif("8" in task_str):
            print("Using learning parameters for CRAFT Task 8: BEFEC")
            test_freq = 400
            test_epi_length = 400
            total_train_step = int(4e5)
            num_random_action = int(8e4)
        elif("11" in task_str):
            print("Using learning parameters for CRAFT Task 11: BEABC Spear")
            test_freq = 400
            test_epi_length = 400
            total_train_step = int(25e4)
            num_random_action = int(5e4)
        else:
            print("Default Learning params being used: Not parameters used in experiments in research paper")
            test_freq = 200
            test_epi_length = 200
            total_train_step = int(2e6)
            num_random_action = int(2e5)
    else:
        print(env + " not supported")
    task = task_str.find(".txt")
    task = task_str[task-2:-4]
    print("\n----------------------")
    print("Task is :", task)
    print("----------------------\n")

    save = True #indicates that we are saving data
    print("Running", rl, "in", env, "using seed", n_seed)
    if save: print("SAVING RESULTS!")
    else:    print("*NOT* SAVING RESULTS!")

    # Setting the learning parameters
    lp = Rod_LearningParameters()
    #below we allow the agent to explore the environment for 2e5 steps before we start training
    lp.set_rm_learning(rm_init_steps=num_random_action, rm_u_max=10, rm_preprocess=True, rm_tabu_size=10000,
                       rm_lr_steps=100, rm_workers=n_workers)
    #below we set the train_steps to 2e6
    lp.set_rl_parameters(gamma=0.9, train_steps=total_train_step, episode_horizon=int(5e3), epsilon=0.1, max_learning_steps=total_train_step)

    #below we determine how often we print results. Right now we print results every 1e4 time steps
    lp.set_test_parameters(test_freq = test_freq,test_epi_length=test_epi_length)

    #below we set learning rate, batch_size and other hyper parameters
    lp.set_deep_rl(lr = 5e-5, learning_starts = 50000, train_freq = 1, target_network_update_freq = 100,
                    buffer_size = 100000, batch_size = 32, use_double_dqn = False, num_hidden_layers = 5, num_neurons = 64)

    # Setting the environment
    env_params = set_environment(env, lp,experiment)

    # Choosing the RL algorithm


    # Running the experiment
    run_lrm_experiments(env_params, lp, rl, n_seed, save,trails=num_trails,task = task) #to see specifics look at run_lrm.py

def set_environment(env,lp,experiment):
    if "active" in env:
        game_type = "officeworld_active"
    elif "office" in env:
        game_type = "officeworld"
    elif "craft" in env:
        game_type ="craftworld"
        map = "../experiments/craft/maps/map_0.map"
        use_tabular_representation=True
        consider_night=False
        return GridWorldParams(game_type="craftworld", file_map=map, movement_noise=0.05,experiment=experiment,use_tabular_representation=use_tabular_representation,consider_night=consider_night)
    return GridWorldParams(game_type=game_type, file_map=None, movement_noise=0.05,experiment=experiment)
if __name__ == "__main__":

    # EXAMPLE: python3 run.py --algorithm="jirp" --world="craft" --map=0 --num_times=1 --show_plots=1 --is_SAT=1

    # Getting params
    algorithms = ["hrl", "jirp", "qlearning", "ddqn",'lrm-qrm','lrm-dqn']
    #lrm is Rodrigo's method. The qrm or dqn specifies how policies are learned in each state of the reward machine

    worlds = ["office_active", "craft", "traffic"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a particular environment.')

    """Change the below line's default to change the type of algorithm"""
    parser.add_argument('--algorithm', default='lrm-qrm', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    """Change the below line's default to change the world (only office world works with LRM at this time) """
    parser.add_argument('--world', default='craft', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    """The below arguements are not uses in LRM"""
    parser.add_argument('--map', default=0, type=int, 
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--num_times', default=10, type=int,
                        help='This parameter indicated which map to use. It must be a number greater or equal to 1')
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument("--show_plots", default=0, help="1 for showing plots throughout the algorithm run, 0 otherwise")
    parser.add_argument("--is_SAT", default=1, help="1 for SAT, 0 for RPNI")
    #parser.add_argument("--trails", default=[1,2,3], help="List of what trails you want to run for LRM")

    args = parser.parse_args()
    """The code below raises errors if an incorrect argument is given"""
    if args.algorithm not in algorithms: raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds: raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not(0 <= args.map <= 10): raise NotImplementedError("The map must be a number between 0 and 10")
    if args.num_times < 1: raise NotImplementedError("num_times must be greater than 0")


    # Running the experiment
    alg_name   = args.algorithm
    world      = args.world
    map_id     = args.map
    num_times  = args.num_times
    show_print = args.verbosity is not None
    show_plots = (int(args.show_plots) == 1)
    is_SAT = (int(args.is_SAT) == 1)
    """The following if statements set variables that do not matter for running LRM (these variables are not passed into the LRM algorithm)"""
    """This variables set the location for hypothesis and test reward machine files"""
    if world == "office" or "office" in world:
        experiment_l = "../experiments/office/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/office/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/office/tests/ground_truth.txt"
            experiment_t = "../experiments/office/tests/ground_truth.txt"
    elif world == "craft":
        experiment_l = "../experiments/craft/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/craft/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/craft/tests/ground_truth.txt"
            experiment_t = "../experiments/craft/tests/ground_truth.txt"
    else:
        experiment_l = "../experiments/traffic/tests/hypothesis_machines.txt"
        experiment_t = "../experiments/traffic/tests/ground_truth.txt"
        if alg_name == "hrl":
            experiment_l = "../experiments/traffic/tests/ground_truth.txt"
            experiment_t = "../experiments/traffic/tests/ground_truth.txt"
    world += "world"


    num_trials = range(num_times)
    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment_l, "num_times: " + str(num_times), show_print)
    run_experiment(world, alg_name, experiment_t, experiment_l, num_times, show_print, show_plots, is_SAT,num_trials)
