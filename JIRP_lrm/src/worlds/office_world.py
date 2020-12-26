from typing import Dict, Tuple

if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')
    is_debug = True
else:
    is_debug = False

from worlds.game_objects import Actions
from worlds.game_objects import *
from worlds.grid_world import GridWorldParams, GridWorld, run_human_agent
import random, math, os
import numpy as np
import random, math, os
from reward_machines.env_reward_machine import EnvRewardMachine
from data_processing import plot_performance,plot_this
"""
Auxiliary class with the configuration parameters that the Game class needs
"""


class OfficeWorldParams():
    def __init__(self):
        pass


class OfficeWorld(GridWorld):

    def __init__(self, params):
        super().__init__(params)
        self.env_game_over = False
        f = open(params.experiment_file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the test attributes
        self.rm_file = eval(lines[1])[0]
        if is_debug: # if we run from officeworld.py we need one more "../"
            self.rm_file = "../" + self.rm_file
        self.rm = EnvRewardMachine(self.rm_file)
        self.current_state = self.get_state()  # get the initial reward machine and MDP state
        self.u1 = self.rm.get_initial_state()
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        #print("THIS IS OFFICE WORLD")

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        #changes depending on the self.rm_file that you use.
        delta_u = {}
        delta_u[(0, 'e')] = 1
        delta_u[(1, 'g')] = 2
        delta_u[(2, 'c')] = 3
        return delta_u

    def get_is_done(self):
        return self.env_game_over

    def execute_action(self, a): #return reward done, done = True when epi is over
        """
        We execute 'action' in the game
        return reward and done
        """
        x = self.agent.i
        y = self.agent.j
        # executing action
        (x,y) = self.xy_MDP_slip(a, 0.9)  # progresses in x-y system
        self.agent = Agent(x,y,Actions)
        u1 = self.u1
        s1 = self.get_state()
        events = self.get_events()  # get conditions of the game
        u2 = self.rm.get_next_state(u1, events)  # get the next state
        s2 = self.get_state()
        reward = self.rm.get_reward(u1, u2, s1, a, s2)  # use the reward machine to generate the rewards
        next_state = s2
        self.u1 = u2
        self.current_state = next_state
        self.env_game_over = self.rm.is_terminal_state(u2)
        done = self.get_is_done()
        return reward, done

    def xy_MDP_slip(self, a, p):
        x = self.agent.i
        y = self.agent.j
        slip_p = [p, (1 - p) / 2, (1 - p) / 2]
        check = random.random()

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check <= slip_p[0]):
            a_ = a

        elif (check > slip_p[0]) & (check <= (slip_p[0] + slip_p[1])):
            if a == 0:
                a_ = 3
            elif a == 2:
                a_ = 1
            elif a == 3:
                a_ = 2
            elif a == 1:
                a_ = 0

        else:
            if a == 0:
                a_ = 1
            elif a == 2:
                a_ = 3
            elif a == 3:
                a_ = 0
            elif a == 1:
                a_ = 2

        action_ = Actions(a_)
        if (x, y, action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                y += 1
            if action_ == Actions.down:
                y -= 1
            if action_ == Actions.left:
                x -= 1
            if action_ == Actions.right:
                x += 1

        self.a_ = a_
        return (x, y)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.a_

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if (self.agent.i,self.agent.j) in self.objects:
            ret += self.objects[(self.agent.i,self.agent.j)]
        return ret

    def get_state(self):
        return self.agent.i * 9 + self.agent.j + 1  # the plus one eliminates the 0 tile. states go from 1 to 108

    # The following methods return different feature representations of the map ------------
    #NOTE only left for the time being (july 8 2020)
    #this method is defined in superclass grid_world.py
    """def get_features(self):
        x, y = self.agent.i,self.agent.j
        N, M = 12, 9
        ret = np.zeros((N, M), dtype=np.float64)
        ret[x, y] = 1
        return ret.ravel()  # from 2D to 1D (use a.flatten() is you want to copy the array)
        """

    def show(self):
        for y in range(8, -1, -1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.up) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()
            for x in range(12):
                if (x, y, Actions.left) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 0:
                    print(" ", end="")
                if (x, y) == (self.agent.i,self.agent.j):
                    print("A", end="")
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)], end="")
                else:
                    print(" ", end="")
                if (x, y, Actions.right) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 2:
                    print(" ", end="")
            print()
            if y % 3 == 0:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.down) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()

                # The following methods create the map ----------------------------------------------

    def _load_map(self, file_map):
        # NOTE: file_map is not used. it is here so that the office world can be a subclass of Grid World
        # Creating the map
        self.map = [[Empty(x, y, label=" ") for y in range(9)] for x in range(12)]  # an empty 12x9 list
        self.objects = {}
        # env.agent = tuple([2, 2])
        # env.coffee = tuple([3, 5])
        # env.init_agent = tuple([2, 2])
        # env.locations = {(1, 1): 'a', (10, 1): 'b', (7, 3): 'c', (7, 4): 'e', (3, 5): 'f', (4, 4): 'g', (1, 8): 'd'}
        # env.mail = tuple([7, 4])
        self.objects[(1, 1)] = "a"
        self.map[1][1] = Empty(1, 1, label="a")  # this line adds stuff to the map
        self.objects[(10, 1)] = "b"
        self.map[10][1] = Empty(10, 1, label="b")
        # self.objects[(10,7)] = "c"
        self.objects[(1, 3)] = "c"
        self.map[1][3] = Empty(1, 3, label="c")
        # self.objects[(1,7)] = "d"
        self.objects[(7, 4)] = "e"  # MAIL
        self.map[7][4] = Empty(7, 4, label="e")
        # self.objects[(8,2)] = "f"  # COFFEE
        self.objects[(3, 5)] = "f"  # COFFEE
        self.map[3][5] = Empty(3, 5, label="f")
        self.objects[(4, 4)] = "g"  # OFFICE
        self.map[4][4] = Empty(4, 4, label="g")
        # Adding walls
        self.forbidden_transitions = set()

        #set up self.map_classes
        self.map_classes = self.get_map_classes()

        # general grid
        for x in range(12):
            for y in [0, 3, 6]:
                self.forbidden_transitions.add((x, y, Actions.down))
                self.forbidden_transitions.add((x, y + 2, Actions.up))
        for y in range(9):
            for x in [0, 3, 6, 9]:
                self.forbidden_transitions.add((x, y, Actions.left))
                self.forbidden_transitions.add((x + 2, y, Actions.right))
        # adding 'doors'
        for y in [1, 7]:
            for x in [2, 5, 8]:
                self.forbidden_transitions.remove((x, y, Actions.right))
                self.forbidden_transitions.remove((x + 1, y, Actions.left))
        for x in [1, 4, 7, 10]:
            self.forbidden_transitions.remove((x, 5, Actions.up))
            self.forbidden_transitions.remove((x, 6, Actions.down))
        for x in [1, 10]:
            self.forbidden_transitions.remove((x, 2, Actions.up))
            self.forbidden_transitions.remove((x, 3, Actions.down))
        # Adding the agent
        self.agent = Agent(2,1,Actions)

        # create self.rooms for gridworld dependency
        """Description of self.rooms
        self.rooms = [ [ ] ....]
        a list of lists. each inner list has two tuples.
        The first tuple corresponds to the bottom left coordinate of a room
        The second tuple corresponds to the top right coordinate of a room
        ie self.rooms = [ [ (0,0), (2,2) ], [(0,3), (2,5)] ] means that there are two rooms; one covers the area 0,0 to 2,2 etc.
        since our world is a 9x12 grid with 12 rooms of equal size (each room is a 3 space by 3 space)
        A formula is used to create self.rooms for office world.
        """
        self.rooms = []  # initialize rooms to be empty; the for loop fills it up
        for y1 in [0, 3, 6]:
            y2 = y1 + 2  # x2 will be 2 when x1 = 0, etc
            for x1 in [0, 3, 6, 9]:
                x2 = x1 + 2
                tuple_1 = (x1, y1)
                tuple_2 = (x2, y2)
                list_1 = [tuple_1, tuple_2]
                self.rooms.append(list_1)
        self.map_locations = []  # list of tuples (room_id, loc_i, loc_j) with all the non-obstacle locations
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if str(self.map[i][j]) != "X":
                    room_id = self._get_room(i, j)
                    self.map_locations.append((room_id, i, j))

    def get_map_classes(self):
        # Returns the string with all the classes of objects that are part of this domain
        # why abcefg, look at _load_map and see what labels I'm giving to the Empty-class objects
        return "abcefg"


    def get_all_events(self):
        return "abcefg"

#Office World Active is the Office world from the active learning paper.
#it is defined as a subclass of Office world because this construction makes it easier to preserve parts of office world that are necessary
#without copying a bunch of code
class OfficeWorldActive(OfficeWorld):
    def __init__(self,params):
        super().__init__(params)
        self.current_state = self.get_state()  # get the initial reward machine and MDP state
        self.u1 = self.rm.get_initial_state()
        #print("THIS IS OFFICE WORLD ACTIVE")

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        #changes depending on the self.rm_file that you use.
        delta_u = {}
        delta_u[(0, 'b')] = 1
        delta_u[(1, 'a')] = 2
        delta_u[(2, 'c')] = 3
        delta_u[(3, 'b')] = 4
        delta_u[(4, 'c')] = 5
        delta_u[(5, 'a')] = 6
        return delta_u

    def _load_map(self, file_map):
        # NOTE: file_map is not used. it is here so that the office world can be a subclass of Grid World
        # Creating the map
        self.map = [[Empty(x, y, label=" ") for y in range(9)] for x in range(12)]  # an empty 12x9 list
        self.objects = {}
        # env.agent = tuple([2, 2])
        # env.coffee = tuple([3, 5])
        # env.init_agent = tuple([2, 2])
        # env.locations = {(1, 1): 'a', (10, 1): 'b', (7, 3): 'c', (7, 4): 'e', (3, 5): 'f', (4, 4): 'g', (1, 8): 'd'}
        # env.mail = tuple([7, 4])
        self.objects[(1, 1)] = "a"
        self.map[1][1] = Empty(1, 1, label="a")  # this line adds stuff to the map
        self.objects[(10, 1)] = "b"
        self.map[10][1] = Empty(10, 1, label="b")
        # self.objects[(10,7)] = "c"
        self.objects[(1, 3)] = "c"
        self.map[1][3] = Empty(1, 3, label="c")
        # Adding walls
        self.forbidden_transitions = set()

        #set up self.map_classes
        self.map_classes = self.get_map_classes()

        # general grid/walls
        for x in range(12):
            for y in [0]:
                self.forbidden_transitions.add((x,y,Actions.down))
            for y in [8]:
                self.forbidden_transitions.add((x,y,Actions.up))
        for y in range(9):
            for x in [0]:
                self.forbidden_transitions.add((x,y,Actions.left))
            for x in [11]:
                self.forbidden_transitions.add((x,y,Actions.right))
        for x in range(12):
             for y in [0,3,6]:
                 self.forbidden_transitions.add((x,y,Actions.down))
                 self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
             for x in [0,3,6,9]:
                 self.forbidden_transitions.add((x,y,Actions.left))
                 self.forbidden_transitions.add((x+2,y,Actions.right))
         # adding 'doors'
        for y in [1,7]:
             for x in [2,5,8]:
                 self.forbidden_transitions.remove((x,y,Actions.right))
                 self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
             self.forbidden_transitions.remove((x,5,Actions.up))
             self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
             self.forbidden_transitions.remove((x,2,Actions.up))
             self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.agent = Agent(2,1,Actions)

        # create self.rooms for gridworld dependency
        """Description of self.rooms
        self.rooms = [ [ ] ....]
        a list of lists. each inner list has two tuples.
        The first tuple corresponds to the bottom left coordinate of a room
        The second tuple corresponds to the top right coordinate of a room
        ie self.rooms = [ [ (0,0), (2,2) ], [(0,3), (2,5)] ] means that there are two rooms; one covers the area 0,0 to 2,2 etc.
        since our world is a 9x12 grid with 12 rooms of equal size (each room is a 3 space by 3 space)
        A formula is used to create self.rooms for office world.
        """
        self.rooms = []  # initialize rooms to be empty; the for loop fills it up
        for y1 in [0, 3, 6]:
            y2 = y1 + 2  # x2 will be 2 when x1 = 0, etc
            for x1 in [0, 3, 6, 9]:
                x2 = x1 + 2
                tuple_1 = (x1, y1)
                tuple_2 = (x2, y2)
                list_1 = [tuple_1, tuple_2]
                self.rooms.append(list_1)
        self.map_locations = []  # list of tuples (room_id, loc_i, loc_j) with all the non-obstacle locations
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if str(self.map[i][j]) != "X":
                    room_id = self._get_room(i, j)
                    self.map_locations.append((room_id, i, j))
    def get_map_classes(self):
            # Returns the string with all the classes of objects that are part of this domain
            # why abcefg, look at _load_map and see what labels I'm giving to the Empty-class objects
            return "abc"

    # play is an old function used for debugging office world; use test_env() instead
    def get_all_events(self):
        return "abc"


# play is an old function used for debugging office world; use test_env() instead
def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w": Actions.up.value, "d": Actions.right.value, "s": Actions.down.value, "a": Actions.left.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../experiments/office/reward_machines/t%d.txt" % i for i in [1, 2, 3, 4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(EnvRewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params)  # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            # print(game.getLTLGoal())
            # Getting action
            print("u:", u1)
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                game.execute_action(str_to_action[a])

                # Getting new state and truth valuation
                s2 = game.get_state()
                events = game.get_true_propositions()
                u2 = rm.get_next_state(u1, events)
                r = rm.get_reward(u1, u2, s1, a, s2)

                # Getting rewards and next states for each reward machine
                rewards, next_states = [], []
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)

                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")

                if game.env_game_over or rm.is_terminal_state(u2):  # Game Over
                    break

                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())


# this method is for debugging. If you create a new method and want to see what it returns, add code here, then click run
def test_the_env():
    experiment = "../../experiments/office/tests/ground_truth.txt"
    params = GridWorldParams(game_type="officeworld", file_map=None, movement_noise=0.05,experiment=experiment)
    game = OfficeWorld(params)
    #x = game.get_features()
    #print(x)
    #print("Length of get features is " + str(len(x)))
    # Print out the map so we can see where the agent is
    game.show()
    # Show the state of the reward machine
    print("State of reward machine is " + str(game.u1))

    # Show the Reward at this state
    print("Reward at THIS STATE is " + str(game.rm.get_reward(game.u1, game.u1)))
    print("State of MDP is " + str(game.current_state))
    # have the variable 'total_reward' keep track of all the rewards given
    total_reward = game.rm.get_reward(game.u1, game.u1)
    print("The total reward given is " + str(total_reward))

    # print(game.get_reward_list())
    # print("Game.getfeaures() returns")
    # print(game.get_features())
    # dictionary to parse input
    act_to_num = {"w": 0, "d": 1, "s": 2, "a": 3}
    done = False  # since the game has started we know that the game is not done
    while not done:
        # ask for user input
        act = input("Action? (w,a,s,d)")
        # Check if the action is valid
        if (act in act_to_num):
            # then do that action
            reward, done = game.execute_action(act_to_num[act])
        else:
            print("Invalid action")
        # Show game map, states and rewards
        game.show()
        """
      print(nxt_state)
      print(str(type(nxt_state)))
      print(reward)
      print(str(type(reward)))
      """
        print("State of reward machine is " + str(game.u1))
        print("State of MDP is " + str(game.current_state))
        print("Reward at THIS STATE is " + str(reward))
        total_reward = total_reward + reward
        print("The total reward given is " + str(total_reward))
        #x = game._get_event_features()
        #print(x)
        #print("game.get_features() returns") #the nxt_state is the output of step. It is the state that the agent currently is in
        #print(game._get_map_features())
        #print(game._get_event_features())

def test_random_action():
    #parameters for testing agent throughout training: 8/7/20
    test_frq = 1000 #how often we test agent 8/7/20
    test_epi_length = 1000 #how long we test agent 8/7/20
    plot_dict = dict() #used for plotting rewards over time from tests 8/7/20
    test_step = 0 #used by plot_dict (variable above) 8/7/20
    #parameters to study while testing
    num_of_suc = 0 #how many times does the agent complete an episode in testing 8.7.20
    time_to_suc = [] #track how long it took agent to complete episode 8.7.20

    params = GridWorldParams(game_type="officeworld", file_map=None, movement_noise=0.05)
    test_env = OfficeWorldActive(params) #environement for testing 8.7.20

    actions = test_env.get_actions()

    for test in range(200): #we test 200 times in run_lrm.py
        test_env = OfficeWorldActive(params) #reset the environment
        test_reward = 0
        test_done = False
        for test_trail in range(test_epi_length): #this is the testing loop
            if not(test_done): #if an episode isn't complete
                act = random.choice(actions) #choose a random action
                test_reward,test_done = test_env.execute_action(act) #execute that action
            else:#if an episode was completed:
                num_of_suc += 1 #increment number of successes but agent to complete an episode 8.7.20
                time_to_suc.append(test_trail) #record how long it took to complete an episode 8.7.20
                break #break out of for loop
        test_step += test_frq #increment test_step by the test_frq
        if test_step in plot_dict.keys(): #this if/else updates plot_dict
            plot_dict[test_step].append(test_reward) #this code was copied from final-i-caps
        else:
            plot_dict[test_step] = [test_reward]
    assert len(plot_dict) == 200 #we check that the agent was tested 2e5/test_frq = 2e5/1e3 = 2e2 = 200 times
    assert len(time_to_suc) == num_of_suc #the number of completed episodes should match how many 'times to completed episodes' we recorded
    import matplotlib.pyplot as plt
    rewards_plot = list()
    prc_25 = list()
    prc_50 = list()
    prc_75 = list()
    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()
    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
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
    plot_performance(steps_plot,prc_25,prc_50,prc_75,"Rewards (percentiles) vs. Time Step",'LRM-qrm 3 symbol OfficeWorld Random Action')
    plot_this(steps_plot,rewards_plot,"Raw Reward vs. Time Step",'LRM-qrm 3 symbol OfficeWorld Random Action')
    plt.plot(time_to_suc,label = "Number of steps to complete episode with random action")
    plt.title("Steps to complete episode\nNumber of Successes = "+str(num_of_suc)+"/200 tests\nAverage number of steps = "+str(np.average(time_to_suc)))
    plt.show()
    adi = input("Continue?")
# This code allow to play a game (for debugging purposes)
# It runs test_env()
if __name__ == '__main__':
    # play()
    #test_env()
    #test_random_action()
    test_the_env()
