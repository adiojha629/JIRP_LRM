# Active Learning
Implements the Active Finite Reward Automaton Inference and Reinforcement Learning (AFRAI-RL) algorithm and Case Study Algorithms

Created by Zhe Xu, Bo Wu, Ivan Gavran, Daniel Neider, Yousef Ahmad, Ufuk Topcu, and Adi Ojha.

RL code modified from Rodrigo Toro Icarte's codes in https://bitbucket.org/RToroIcarte/qrm/src/master/.

The following algorithms are present:<br>
-Active Finite Reward Automaton Inference and Reinforcement Learning (AFRAI-RL) in active16 folder<br>
-Joint Inference of Reward Machines and Policies (JIRP) in JIRP_code folder<br>
-Learned Reward Machines (qrm implementation) in JIRP_LRM folder

# Prerequisites:
Ubuntu or Linux Operating System: Code runs bash commands that are not native to Windows OS (ie. ls not a valid windows command line operation)

## Tensorflow Version 2
The code may issue warnings about Tensorflow version 2. Please ignore these, as they will be addressed in a later update of the code.<br>
Tensorflow Version 1 will cause errors.

# How to run examples
## AFRAI-RL
In a command line, navigate to *JIRP_LRM/active16/src*. Use this command:<br>
``` python3 run1.py --algorithm="aqrm" --world=<environment> --map=0 --num_times=<number of trials> ``` 
## JIRP-SAT
In a command line, navigate to *JIRP_LRM/JIRP_code/src*. Use this command:<br>
``` python3 run1.py --algorithm="jirp" --world=<environment> --map=0 --num_times=<number of trials> ``` 
## LRM-QRM
In a command line, navigate to *JIRP_LRM/JIRP_lrm/src*. Use this command:<br>
``` python3 run1.py --algorithm="lrm-qrm" --world=<environment> --map=0 --num_times=<number of trials> ``` 
<br><br>
The ```run1.py``` code receives 4 parameters: The RL algorithm to use (which might be "aqrm", "jirp", or "lrm-qrm", the first and second of which are our methods), the environment (which might be "office", "craft", or "traffic"), the map (which is integer 0), and the number of independent trials to run per map.
<br>
## Change the Task
In order to change the task being performed, move to the corresponding folder from the *experiments* folder and change the task index specified in the ground truth file found in the *tests* folder. For example, in order to run task 2 from the craft world, set the index (on line 2) between the square brackets as indicated in 'experiments/craft/tests/ground_truth.txt' before running *run1.py*:

    ["../experiments/craft/reward_machines/t%d.txt"%i for i in [2]]  # tasks

All results are saved in 'src/automata_learning_utils/data'.

## Acknowledgments

Our implementation is modified from the QRM codes provided by (https://bitbucket.org/RToroIcarte/qrm). The benchmarks are based on the codes provided by [OpenAI](https://github.com/openai/baselines).
