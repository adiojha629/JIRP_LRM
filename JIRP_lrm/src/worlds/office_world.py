if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import Actions
from worlds.game_objects import *
from worlds.grid_world import GridWorldParams, GridWorld, run_human_agent
import random, math, os
import numpy as np
import random, math, os
import numpy as np
from reward_machines.env_reward_machine import EnvRewardMachine

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class OfficeWorldParams():
    def __init__(self):
        self.grid_world_params = GridWorldParams(self,game_type = "officeworld", file_map = None, movement_noise = 0.05)

class OfficeWorld(GridWorld):

    def __init__(self, params):
        super().__init__(GridWorldParams(self,game_type = "office", file_map = None, movement_noise = 0.05))
        self._load_map()
        self.env_game_over = False
        self.rm_file = "../../experiments/office/reward_machines/t1.txt"
        self.env_rm = EnvRewardMachine(self.rm_file)
        self.current_state = self.get_state()#get the initial reward machine and MDP state
        self.u1 = self.env_rm.get_initial_state()
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

    def get_perfect_rm(self):
        # NOTE: This is used for debugging purposes and to compute the expected reward of an optimal policy
        delta_u = {}
        delta_u[(0, 'e')] = 1
        delta_u[(1, 'g')] = 2
        delta_u[(2, 'c')] = 3
        return delta_u
    def get_is_done(self):
        return self.env_game_over
    def execute_action(self, a):
        """
        We execute 'action' in the game
        return reward and done
        """
        x,y = self.agent
        # executing action
        self.agent = self.xy_MDP_slip(a,0.9) # progresses in x-y system
        u1 = self.u1
        s1 = self.get_state()
        events = self.get_events()#get conditions of the game
        u2 = self.env_rm.get_next_state(u1, events)#get the next state
        s2 = self.get_state()
        reward = self.env_rm.get_reward(u1,u2,s1,a,s2)#use the reward machine to generate the rewards
        next_state = s2
        self.u1 = u2
        self.current_state = next_state
        done = self.get_is_done()
        return reward,done


    def xy_MDP_slip(self,a,p):
        x,y = self.agent
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

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
        if (x,y,action_) not in self.forbidden_transitions:
            if action_ == Actions.up:
                y+=1
            if action_ == Actions.down:
                y-=1
            if action_ == Actions.left:
                x-=1
            if action_ == Actions.right:
                x+=1

        self.a_ = a_
        return (x,y)

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
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_state(self):
        return self.agent[0]*9 + self.agent[1] + 1#the plus one eliminates the 0 tile. states go from 1 to 108

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        x,y = self.agent
        N,M = 12,9
        ret = np.zeros((N,M), dtype=np.float64)
        ret[x,y] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)


    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    # The following methods create the map ----------------------------------------------
    def _load_map(self):
        # Creating the map
        self.objects = {}
        #env.agent = tuple([2, 2])
        #env.coffee = tuple([3, 5])
        #env.init_agent = tuple([2, 2])
        #env.locations = {(1, 1): 'a', (10, 1): 'b', (7, 3): 'c', (7, 4): 'e', (3, 5): 'f', (4, 4): 'g', (1, 8): 'd'}
        #env.mail = tuple([7, 4])
        self.objects[(1,1)] = "a"
        self.objects[(10,1)] = "b"
        #self.objects[(10,7)] = "c"
        self.objects[(1, 3)] = "c"
        #self.objects[(1,7)] = "d"
        self.objects[(7,4)] = "e"  # MAIL
        #self.objects[(8,2)] = "f"  # COFFEE
        self.objects[(3,5)] = "f"  # COFFEE
        self.objects[(4,4)] = "g"  # OFFICE

        # Adding walls
        self.forbidden_transitions = set()
        # for x in range(12):
        #     for y in [0]:
        #         self.forbidden_transitions.add((x,y,Actions.down))
        #     for y in [8]:
        #         self.forbidden_transitions.add((x,y,Actions.up))
        # for y in range(9):
        #     for x in [0]:
        #         self.forbidden_transitions.add((x,y,Actions.left))
        #     for x in [11]:
        #         self.forbidden_transitions.add((x,y,Actions.right))
        # general grid
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
        self.agent = (2,1)

#play is an old function used for debugging office world; use test_env() instead
def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../experiments/office/reward_machines/t%d.txt"%i for i in [1,2,3,4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(EnvRewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params) # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            #print(game.getLTLGoal())
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
                r = rm.get_reward(u1,u2,s1,a,s2)
                
                # Getting rewards and next states for each reward machine
                rewards, next_states = [],[]
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)
                
                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")
                
                if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                    break 
                
                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())

#this method is for debugging. If you create a new method and want to see what it returns, add code here, then click run
def test_env():
    params = OfficeWorldParams()
    game = OfficeWorld(params)
    #Print out the map so we can see where the agent is
    game.show()
    #Show the state of the reward machine
    print("State of reward machine is " + str(game.u1))

    #Show the Reward at this state
    print("Reward at THIS STATE is "+ str(game.env_rm.get_reward(game.u1,game.u1)))
    print("State of MDP is " + str(game.current_state))
    #have the variable 'total_reward' keep track of all the rewards given
    total_reward = game.env_rm.get_reward(game.u1,game.u1)
    print("The total reward given is " + str(total_reward))

    #print(game.get_reward_list())
    print("Game.getfeaures() returns")
    print(game.get_features())
    #dictionary to parse input
    act_to_num = {"w":0,"d":1,"s":2,"a":3}
    done = False #since the game has started we know that the game is not done
    while not done:
      #ask for user input
      act = input("Action? (w,a,s,d)")
      #Check if the action is valid
      if(act in act_to_num):
        #then do that action
        reward, done = game.execute_action(act_to_num[act])
      else:
        print("Invalid action")
      #Show game map, states and rewards
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
      print("game.get_features() returns") #the nxt_state is the output of step. It is the state that the agent currently is in
      print(game.get_features())
# This code allow to play a game (for debugging purposes)
#It runs test_env()
if __name__ == '__main__':
    #play()
    test_env()
