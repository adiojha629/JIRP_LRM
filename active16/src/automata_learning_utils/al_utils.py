import argparse
import pdb
import subprocess
from .dfa import DFA
import os, sys
#from subprocess import Popen, PIPE

#process = Popen(args,stdout=PIPE,stdin=PIPE,universal_newlines=True, bufsize=1)
#print(process.stdout.readline())
#process.stdin.write('2\n')

def learn_automaton(traces_filename,
                    automaton_learning_program = None,
                    output_reward_machine_filename=None):


    dirname = os.path.abspath(os.path.dirname(__file__))

    if output_reward_machine_filename == None:
        output_reward_machine_filename = os.path.join(dirname, "data/rm.txt")


    if automaton_learning_program == None:

        automaton_learning_program = os.path.join(dirname, "../examples/online")


    output_filename = os.path.join(dirname, "data/automaton.txt")
    output_visualization_filename = os.path.join(dirname, "data/hypothesis.dot")
    #subprocess.run([automaton_learning_program, traces_filename, output_filename, output_visualization_filename])
    #subprocess.run(["xdot", output_visualization_filename])

    dfa = DFA(output_filename)
    dfa.export_as_reward_automaton(output_reward_machine_filename)
    return os.path.abspath(output_visualization_filename)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_filename")

    args = parser.parse_args()
    learn_automaton(args.traces_filename)


if __name__ == '__main__':
    main()
