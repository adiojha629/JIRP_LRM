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
from qrm.learning_params import LearningParameters



def get_params_craft_world(experiment,step_unit,total_num_steps):
    #step_unit = 600

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
    learning_params.lr = 5e-5
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 32
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 50000

    # Tabular case
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 5
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = total_num_steps
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


def get_params_office_world(experiment,step_unit,total_num_steps):
    #step_unit = 800 value set above

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq =step_unit
    testing_params.num_steps = step_unit  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.memory_size = 200
    learning_params.buffer_size = 10
    learning_params.relearn_period = 30
    learning_params.enter_loop = 10
    learning_params.lr = 4e-4 #5e-5 Changed to 4e-4 on 10.10.20 by Aditya Ojha
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 1
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 50000

    # Tabular case
    learning_params.tabular_case = False  
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 5
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = total_num_steps
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
    learning_params.lr = 1 #for q-learning the a separate parameter, which is 0.8
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
    tester = Tester(learning_params, testing_params, experiment)


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


def run_experiment(world, alg_name, experiment_known, experiment_learned, num_times, show_print, show_plots, is_SAT):
    if world == 'officeworld':
        f = open(experiment_known)
        lines = [l.rstrip() for l in f]
        f.close()
        task_str = eval(lines[1])[0]
        task = task_str.find(".txt")
        task = task_str[task-2:-4] #get task number
        step_unit = total_num_steps = num_random_action = 0 #variables that vary based on task
        if("7" in task): #check if running abac task
            print("Using learning parameters for Task 7: ABAC")
            step_unit = 200
            total_num_steps = int(1e6)
        elif("9" in task):
            print("Using learning parameters for Task 9: BCABCA")
            step_unit = 800
            total_num_steps = int(2e6)
        elif("10" in task):
            print("Using learning parameters for Task 10: CBABCA")
            step_unit = 800
            total_num_steps = int(6e6)
        else:
            print("Default Learning params being used: Not parameters used in experiments in research paper")
            step_unit = 800
            total_num_steps = int(2e6)
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_office_world(experiment_known,step_unit,total_num_steps)
        testing_params, learning_params, tester_l, curriculum = get_params_office_world(experiment_learned,step_unit,total_num_steps)
        ## allows for 2 sets of testers/curricula: one for previously known (ground truth) and one for learned info
    if world == 'craftworld':
        f = open(experiment_known)
        lines = [l.rstrip() for l in f]
        f.close()
        task_str = eval(lines[2])[0]
        task = task_str.find(".txt")
        task = task_str[task-2:-4] #get task number
        step_unit = total_num_steps = num_random_action = 0 #variables that vary based on task
        if("7" in task): #check if running abac task
            print("Using learning parameters for Task 7: ABAC")
            step_unit = 400
            total_num_steps = int(4e5)
        elif("9" in task):
            print("Using learning parameters for Task 9: BCABCA")
            step_unit = 600
            total_num_steps = int(6e5)
        elif("6" in task):
            print("Using learning params for Sword!")
            step_unit = 400
            total_num_steps = int(4e5)
        elif("8" in task):
            print("Using learning params for befec!")
            step_unit = 400
            total_num_steps = int(4e5)
        elif("11" in task):
            print("Using learning params for BEABC spear!")
            step_unit = 400
            total_num_steps = int(25e4)
        else:
            print("Default Learning params being used: Not parameters used in experiments in research paper")
            step_unit = 600
            total_num_steps = int(6e5)
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_craft_world(experiment_known,step_unit,total_num_steps)
        testing_params, learning_params, tester_l, curriculum = get_params_craft_world(experiment_learned,step_unit,total_num_steps)
    if world == 'trafficworld':
        testing_params_k, learning_params_k, tester, curriculum_k = get_params_traffic_world(experiment_known)
        testing_params, learning_params, tester_l, curriculum = get_params_traffic_world(experiment_learned)
    #if world == 'waterworld':
        #testing_params_k, learning_params_k, tester, curriculum_k = get_params_water_world(experiment_known)
        #testing_params, learning_params, tester_l, curriculum = get_params_water_world(experiment_learned)

    if alg_name == "ddqn":
        tester = TesterPolicyBank(learning_params, testing_params, experiment_known)
        run_dqn_experiments(alg_name, tester, curriculum, num_times, show_print)

    if alg_name == "hrl":
        if world == 'craftworld':
            testing_params, learning_params, tester, curriculum = get_params_craft_world("../experiments/craft/tests/ground_truth.txt")
        elif world == 'officeworld':
            testing_params, learning_params, tester, curriculum = get_params_office_world("../experiments/office/tests/ground_truth.txt",step_unit,total_num_steps)
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


if __name__ == "__main__":

    # EXAMPLE: python3 run.py --algorithm="jirp" --world="craft" --map=0 --num_times=1 --show_plots=1 --is_SAT=1

    # Getting params
    algorithms = ["hrl", "jirp", "qlearning", "ddqn",'qrm']
    worlds     = ["office", "craft", "traffic"]

    parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithm', default='jirp', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='craft', type=str,
                        help='This parameter indicated which world to solve. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int, 
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--num_times', default=10, type=int,
                        help='This parameter indicated which map to use. It must be a number greater or equal to 1')
    parser.add_argument("--verbosity", help="increase output verbosity")
    parser.add_argument("--show_plots", default=0, help="1 for showing plots throughout the algorithm run, 0 otherwise")
    parser.add_argument("--is_SAT", default=1, help="1 for SAT, 0 for RPNI")
 

    args = parser.parse_args()
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

    if world == "office":
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


    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment_l, "num_times: " + str(num_times), show_print)
    run_experiment(world, alg_name, experiment_t, experiment_l, num_times, show_print, show_plots, is_SAT)