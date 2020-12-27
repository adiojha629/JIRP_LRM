# Iterative automata learning for improving reinforcement learning

This project studies how learning an automaton presenting the temporal logic of rewards might help the reinforcement learning process.
The automaton learning happens simultaneously to reinforcement learning.

Created by Zhe Xu, Bo Wu, Ivan Gavran, Daniel Neider, Yousef Ahmad and Ufuk Topcu.

RL code modified from Rodrigo Toro Icarte's codes in https://bitbucket.org/RToroIcarte/qrm/src/master/.


## QRM

Reward machines are a type of finite state machine that supports the specification of reward functions while exposing reward function structure to the learner and supporting decomposition — and how to use them to speed up learning of optimal policies. Our approach, called Q-Learning for Reward Machines (QRM), decomposes the reward machine and uses off-policy q-learning to simultaneously learn subpolicies for the different components. A detailed description of Reward Machines and QMR can be found in the following paper ([link](http://proceedings.mlr.press/v80/icarte18a.html)):

    @inproceedings{tor-etal-icml18,
        author = {Toro Icarte, Rodrigo and Klassen, Toryn Q. and Valenzano, Richard and McIlraith, Sheila A.},
        title     = {Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning},
        booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
        year      = {2018},
        note      = {2112--2121}
    }

This code is meant to be a clean and usable version of our approach. If you find any bugs or have questions about it, please let us know. We'll be happy to help you!


## Installation instructions

You might clone this repository by running:

    (after publish github)

QRM requires [Python3.5](https://www.python.org/) with three libraries: [numpy](http://www.numpy.org/), [tensorflow](https://www.tensorflow.org/), and (optionally) [pygame](https://www.pygame.org/news). We use pygame to visualize the Water World environment, but it is not required by QRM or any of our baselines.


## Running examples

To run our method and our two baselines, move to the *src* folder and execute *run1.py*. This code receives 4 parameters: The RL algorithm to use (which might be "qlearning", "hrl", or "aqrm", the last of which is our method), the environment (which might be "office", "craft", or "traffic"), the map (which is integer 0), and the number of independent trials to run per map. For instance, the following command runs AQRM one time over map 0 of the craft environment:

    python3 run1.py --algorithm="aqrm" --world="craft" --map=0 --num_times=1

In order to change the task being performed, move to the corresponding folder from the *experiments* folder and change the task index specified in the ground truth file found in the *tests* folder. For example, in order to run task 2 from the craft world, set the index (on line 2) between the square brackets as indicated in 'experiments/craft/tests/ground_truth.txt' before running *run1.py*:

    ["../experiments/craft/reward_machines/t%d.txt"%i for i in [2]]  # tasks

All results are saved in 'src/automata_learning_utils/data'.


## Acknowledgments

Our implementation is modified from the QRM codes provided by (https://bitbucket.org/RToroIcarte/qrm). The benchmarks are based on the codes provided by [OpenAI](https://github.com/openai/baselines).
