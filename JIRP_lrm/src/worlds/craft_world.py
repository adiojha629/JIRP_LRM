if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')
    in_debug = True
else:
    in_debug = False

from worlds.game_objects import *
import random, math, os
import numpy as np
import random
#additional libraries needed for LRM compabitility
from reward_machines.env_reward_machine import EnvRewardMachine
from worlds.grid_world import GridWorldParams, GridWorld, run_human_agent

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class CraftWorldParams:
    def __init__(self, file_map, use_tabular_representation, consider_night, movement_noise = 0):
        self.file_map     = file_map
        self.use_tabular_representation = use_tabular_representation
        self.movement_noise = movement_noise
        self.consider_night = consider_night
class CraftWorld(GridWorld):

    def __init__(self, params):
        print("This is Minecraft World")
        super().__init__(params)
        self.params = params
        self._load_map(params.file_map)
        self.movement_noise = params.movement_noise
        self.env_game_over = False
        # Adding day and night if need it
        self.consider_night = params.consider_night
        self.hour = 12
        if self.consider_night:
            self.sunrise = 5
            self.sunset  = 21
        #code added for LRM compatibility:
        f = open(params.experiment_file)
        lines = [l.rstrip() for l in f]
        f.close()
        # setting the test attributes
        self.rm_file = eval(lines[2])[0] #get the reward machine that defines the task 8.12.20
        if in_debug: #look at line 1's if-else structure for details
            self.rm_file = "../" + self.rm_file
        self.rm = EnvRewardMachine(self.rm_file) #make that reward machine 8.12.20
        self.current_state = self.get_state()  # get the initial reward machine and MDP state
        self.u1 = self.rm.get_initial_state()
        self.actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent
        self.hour = (self.hour + 1)%24

        # MDP
        p = 0.9
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        if (check<=slip_p[0]):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
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
        self.a_ = a_

        # Getting new position after executing action
        ni,nj = self._get_next_position(action_, self.movement_noise)
        
        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni,nj)
        #code for LRM compatibility
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
        done = self.env_game_over
        return reward, done

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    def _get_next_position(self, action, movement_noise):
        """
        Returns the position where the agent would be if we execute action
        """
        agent = self.agent
        ni,nj = agent.i, agent.j

        # without jumping
        direction = action
        cardinals = set([Actions.up, Actions.down, Actions.left, Actions.right])
        if direction in cardinals and random.random() < movement_noise:
            direction = random.choice(list(cardinals - set([direction])))
        

        # OBS: Invalid actions behave as NO-OP
        if direction == Actions.up   : ni-=1
        if direction == Actions.down : ni+=1
        if direction == Actions.left : nj-=1
        if direction == Actions.right: nj+=1

        
        return ni,nj

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.agent.get_actions()

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.a_


    def _is_night(self):
        return not(self.sunrise <= self.hour <= self.sunset)

    def _steps_before_dark(self):
        if self.sunrise - 1 <= self.hour <= self.sunset:
            return 1 + self.sunset - self.hour
        return 0 # it is night

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        # adding the is_night proposition
        if self.consider_night and self._is_night():
            ret += "n"
        return ret

    # The following methods return different feature representations of the map ------------
    #methods commented out b/c get features defined in gridworld class
    '''def get_features(self):
        # if self.params.use_tabular_representation:
        #     return self._get_features_one_hot_representation()
        # return self._get_features_manhattan_distance()
        return self._get_features_one_hot_representation()


    def _get_features_manhattan_distance(self):
        # map from object classes to numbers
        class_ids = self.class_ids #{"a":0,"b":1}
        N,M = self.map_height, self.map_width
        ret = []
        for i in range(N):
            for j in range(M):
                obj = self.map_array[i][j]
                if str(obj) in class_ids:
                    ret.append(self._manhattan_distance(obj))
        
        # Adding the number of steps before night (if need it)
        if self.consider_night:
            ret.append(self._steps_before_dark())

        return np.array(ret, dtype=np.float64)
'''

    """
    Returns the Manhattan distance between 'obj' and the agent
    """
    def _manhattan_distance(self, obj):
        return abs(obj.i - self.agent.i) + abs(obj.j - self.agent.j)

    """
    Returns a one-hot representation of the state (useful for the tabular case)
    """
    #method commented out b/c get features defined in gridworld class
    '''def _get_features_one_hot_representation(self):
        if self.consider_night:
            N,M,T = self.map_height, self.map_width, self.sunset - self.sunrise + 3
            ret = np.zeros((N,M,T), dtype=np.float64)
            ret[self.agent.i,self.agent.j, self._steps_before_dark()] = 1
        else:
            N,M = self.map_height, self.map_width
            ret = np.zeros((N,M), dtype=np.float64)
            ret[self.agent.i,self.agent.j] = 1
        return ret.ravel() # from 3D to 1D (use a.flatten() is you want to copy the array)'''

    # The following methods create a string representation of the current state ---------
    """
    Prints the current map
    """
    def show_map(self):
        print(self.__str__())
        if self.consider_night:
            print("Steps before night:", self._steps_before_dark(), "Current time:", self.hour)

    def __str__(self):
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i,j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if(i > 0):
                r += "\n"
            r += s
        return r
    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        #changes depending on the self.rm_file that you use.
        #I have not implemented this correctly, but it doesn't affect performance of LRM
        delta_u = {}
        delta_u[(0, 'e')] = 1
        delta_u[(1, 'g')] = 2
        delta_u[(2, 'c')] = 3
        return delta_u
    def get_map_classes(self):#used in grid world
            # Returns the string with all the classes of objects that are part of this domain
            # why abcefg, look at _load_map and see what labels I'm giving to the Empty-class objects
            return "abcef"
    def get_all_events(self):#same as get_map_classes, just different names
        return self.get_map_classes()
    # The following methods create the map ----------------------------------------------
    def _load_map(self,file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map (no monsters and no agent)
                - e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
            - self.agent: is the agent!
            - self.map_height: number of rows in every room 
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.map_array = []
        self.class_ids = {} # I use the lower case letters to define the features
        f = open(file_map)
        i,j = 0,0
        for l in f:
            # I don't consider empty lines!
            if(len(l.rstrip()) == 0): continue
            
            # this is not an empty line!
            row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i,j,label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":  entity = Empty(i,j)
                if e == "X":    entity = Obstacle(i,j)
                if e == "A":    self.agent = Agent(i,j,actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])
        #needed for grid world
        self.map = [[Empty(x, y, label=" ") for y in range(self.map_height)] for x in range(self.map_width)]  # an empty 12x9 list
        for i in range(len(self.map_array)):
            for j in range(len(self.map_array[i])):
                entity = str(self.map_array[i][j])
                if not("X" in entity or "A" in entity):
                    #then the entity must be a empty, lets check the label
                    if self.map_array[i][j].label != " ": #if the label is not " " then we put this entity in self.map
                        self.map[i][j] = self.map_array[i][j]
        #set up self.map_classes
        self.map_classes = self.get_map_classes()
        """Description of self.rooms
        self.rooms = [ [ ] ....]
        a list of lists. each inner list has two tuples.
        The first tuple corresponds to the bottom left coordinate of a room
        The second tuple corresponds to the top right coordinate of a room
        ie self.rooms = [ [ (0,0), (2,2) ], [(0,3), (2,5)] ] means that there are two rooms; one covers the area 0,0 to 2,2 etc.
        since our world is a 9x12 grid with 12 rooms of equal size (each room is a 3 space by 3 space)
        A formula is used to create self.rooms for office world.
        
        Since craft world has only one room (the entire board) this is pretty simple
        """
        self.rooms = [[(0,0),(self.map_height-1, self.map_width-1)]]  # initialize rooms to be empty; the for loop fills it up
        self.map_locations = []  # list of tuples (room_id, loc_i, loc_j) with all the non-obstacle locations again needed for grid world 8.12.2020
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if str(self.map[i][j]) != "X":
                    room_id = self._get_room(i, j)
                    self.map_locations.append((room_id, i, j))


def play(params, task, max_time):
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    # play the game!
    game = CraftWorld(params)        
    rm = RewardMachine(task) 
    s1 = game.get_state()
    u1 = rm.get_initial_state()
    for t in range(max_time):
        # Showing game
        game.show_map()
        #print(game.get_features())
        #print(game.get_features().shape)
        #print(game._get_features_manhattan_distance())
        acts = game.get_actions()
        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            game.execute_action(str_to_action[a])

            s2 = game.get_state()
            events = game.get_true_propositions()
            u2 = rm.get_next_state(u1, events)
            reward = rm.get_reward(u1,u2,s1,a,s2)

            if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                break 
            
            s1, u1 = s2, u2
        else:
            print("Forbidden action")
    game.show_map()
    return reward

def test_the_env():
    map = "../../experiments/craft/maps/map_0.map"
    experiment = "../../experiments/craft/tests/ground_truth.txt"
    use_tabular_representation=True
    consider_night=False
    params = GridWorldParams(game_type="craftworld", file_map=map, movement_noise=0.05,experiment=experiment,use_tabular_representation=use_tabular_representation,consider_night=consider_night)
    game = CraftWorld(params)
    x = game.get_features()
    print(x)
    print("Length of get features is " + str(len(x)))
    # Print out the map so we can see where the agent is
    game.show_map()
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
        game.show_map()
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
        x = game._get_event_features()
        print(x)
        print("game.get_features() returns") #the nxt_state is the output of step. It is the state that the agent currently is in
        print(game._get_map_features())
        print(game._get_event_features())


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    test_the_env()
