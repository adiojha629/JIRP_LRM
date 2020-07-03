from worlds.game_objects import *
import random, math, os
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class GridWorldParams:
    def __init__(self, game_type, file_map, movement_noise):
        self.game_type      = game_type
        self.file_map       = file_map
        self.movement_noise = movement_noise

class GridWorld:

    def __init__(self, params):
        self._load_map(params.file_map)
        self.movement_noise = params.movement_noise
        self.u  = 0
        self.rm = self.get_perfect_rm()

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent

        # indicating that the agent is leaving some position
        # (this is useful for getting rewarded after eating a cookie)
        self.map[agent.i][agent.j].leaving()

        # Getting new position after executing action
        ni,nj = self._get_next_position(action, self.movement_noise)

        # Interacting with the objects that is in the next position
        action_succeeded = self.map[ni][nj].interact(agent, action)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni,nj)

        # returns the reward and whether the game is over
        reward, done = self._get_reward_and_gameover()

        # updating the RM state
        o2_events = self.get_events()
        if (self.u,o2_events) in self.rm:
            self.u = self.rm[(self.u,o2_events)]

        return reward, done

    def get_state(self):
        return self.get_events()

    def get_location(self):
        # this auxiliary method allows to keep track of the agent's movements
        return self.agent.i, self.agent.j

    def _get_next_position(self, action, movement_noise):
        """
        Returns the position where the agent would be if we execute action
        """
        agent = self.agent
        ni,nj = agent.i, agent.j

        # without jumping
        direction = action
        if random.random() < movement_noise:
            cardinals = set(self.get_actions())
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

    def get_events(self):
        # Returns the string with the propositions that are True in this state
        raise NotImplementedError("To be implemented")

    def get_map_classes(self):
        # Returns the string with all the classes of objects that are part of this domain
        raise NotImplementedError("To be implemented")

    def get_all_events(self):
        # Returns a string with all the possible events that may occur in the environment
        raise NotImplementedError("To be implemented")

    def _get_reward_and_gameover(self):
        # returns the reward and whether the game is over
        raise NotImplementedError("To be implemented")

    # The following methods create a string representation of the current state ---------
    """
    Prints the current map
    """
    def show_map(self):
        print(self.__str__())
        print("A-Pos:", self.agent.i, self.agent.j)
        print("Room:", self._get_room(self.agent.i,self.agent.j))
        print("Optimal:", self.get_optimal_action())

    def __str__(self):
        agent_room = self._get_room(self.agent.i, self.agent.j)
        r = ""
        for i in range(len(self.map)):
            s = ""
            for j in range(len(self.map[i])):
                if agent_room != self._get_room(i, j):
                    s += " "
                else:
                    if self.agent.idem_position(i,j): s += str(self.agent)
                    else: s += str(self.map[i][j])
            if j > 0: r += "\n"
            r += s
        return r

    def _get_room(self, i, j):
        for room_id in range(len(self.rooms)):
            r = self.rooms[room_id]
            if r[0][0] <= i <= r[1][0] and r[0][1] <= j <= r[1][1]:
                return room_id
        return None

    def _get_features_pos_and_dims(self):
        raise NotImplementedError("To be implemented")

    def _get_map_features(self):
        n_loc = len(self.map_locations) #how many map locations
        n_classes = len(self.map_classes) + 2 # the extra 2 classes are empty and the agent
        map_features = np.zeros((n_loc,n_classes), dtype=np.float64) #create array of dimension map locations by number of classes
        agent_room = self._get_room(self.agent.i, self.agent.j)
        for i in range(len(self.map_locations)):
            room_id, loc_i, loc_j = self.map_locations[i] #each map location has an id, and an x,y coordinate
            if agent_room == room_id: #if the agent is in the room
                map_features[i,0] = 1.0 # the agent can observe this location
                if (loc_i, loc_j) == (self.agent.i, self.agent.j):
                    map_features[i,1] = 1.0 # the agent is at this location
                obj_type = str(self.map[loc_i][loc_j]).lower()
                if obj_type in self.map_classes:
                    map_features[i,self.map_classes.index(obj_type)+2] = 1.0
        return map_features

    def _get_event_features(self):
        all_events = self.get_all_events()
        n_events = len(all_events)
        event_features = np.zeros(n_events, dtype=np.float64)
        detected_events = self.get_events()
        for i in range(n_events):
            event_features[i] = float(detected_events.count(all_events[i]))
        return event_features

    def get_features(self):
        # Creating the map features
        map_features = self._get_map_features()

        # Adding the event detectors
        event_features = self._get_event_features()

        # returning the concatenation
        return np.concatenate((map_features,event_features), axis=None)

    def _load_map(self, file_map):

        # contains all the actions that the agent can perform
        actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value]
        # loading the map
        self.max_i = 0
        self.max_j = 0
        self.map = []
        f = open(file_map)
        for l in f:
            # I don't consider empty lines!
            if len(l.rstrip()) == 0:
                continue

            # adding list of rooms
            if "Rooms:" in l:
                self.rooms = eval(l.rstrip().replace("Rooms: ",""))
                continue

            # this line is part of a room!
            row = []
            for e in l.replace("\n",""):
                i,j = len(self.map), len(row)
                if e in " A": entity = Empty(i,j)
                if e in "X?.": entity = Obstacle(i,j,label=e)
                if e in "abcdefghij": # klmnopqrstuvwxyz are used for the other objects
                    entity = Empty(i,j,label=e)
                if e == "K": entity = Key(i,j)
                if e == "D": entity = Door(i,j)
                if e == "B": entity = Buttom(i,j)
                if e == "T": entity = CookieButtom(i,j)
                if e == "C": entity = Cookie(i,j)
                if e == "A": self.agent = Agent(actions,i,j)
                row.append(entity)
                self.max_i = max([self.max_i, i])
                self.max_j = max([self.max_j, j])
            self.map.append(row)
        f.close()

        # information for the feature representation of the maps
        self.map_locations = [] # list of tuples (room_id, loc_i, loc_j) with all the non-obstacle locations
        self.map_classes = self.get_map_classes()
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if str(self.map[i][j]) != "X":
                    room_id = self._get_room(i, j)
                    self.map_locations.append((room_id, i, j))

    def get_perfect_rm(self):
        # HACK: Returns the transitions for a perfect rm for this domain
        #       This is used for debugging purposes and to compute the expected reward of an optimal policy
        raise NotImplementedError("To be implemented")

    def get_optimal_action(self):
        # HACK: Returns the best possible action given current state
        #       This is used for debugging purposes and to compute the expected reward of an optimal policy
        raise NotImplementedError("To be implemented")

    def _go_to_room(self, room):
        # HACK: Returns the best action towards reaching location (x,y)
        #       This is used for debugging purposes and to compute the expected reward of an optimal policy
        """
        R0: north
        R1: hallway
        R2: south
        R3: East
        """
        i,j = self.agent.i, self.agent.j
        room_agent  = self._get_room(i,j)
        if room_agent == 1:
            if room == 0:
                if j != 2: return Actions.left
                return Actions.up
            elif room == 2:
                if j != 2: return Actions.left
                return Actions.down
            elif room == 3:
                if i < 8:
                    return Actions.down
                if i > 8:
                    return Actions.up
                return Actions.right
            else:
                assert False, "ERROR!"
        if room_agent == 0:
            if room == 1:
                if i == 2: return Actions.down
                if j < 2: return Actions.right
                if j > 2: return Actions.left
                return Actions.down
            assert False, "ERROR!"
        if room_agent == 2:
            if room == 1:
                if i == 14: return Actions.up
                if j < 2: return Actions.right
                if j > 2: return Actions.left
                return Actions.up
            assert False, "ERROR!"
        if room_agent == 3:
            if room == 1:
                if i < 8: return Actions.down
                if i > 8: return Actions.up
                return Actions.left
            assert False, "ERROR!"

def run_human_agent(game, max_time):

    # commands
    str_to_action = {"w":Actions.up.value,"d":Actions.right.value,"s":Actions.down.value,"a":Actions.left.value}
    # play the game!
    trace = [(game.get_events(),0.0)]
    rm = game.get_perfect_rm()
    u1 = 0
    for t in range(max_time):
        # Showing game
        game.show_map()
        print("RM state:", u1)

        if trace[-1][1] > 0: print("Last reward:", trace[-1][1])
        acts = game.get_actions()
        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            reward, is_done = game.execute_action(str_to_action[a])
            events = game.get_events()
            trace.append((events,reward))
            if is_done: # Game Over
                break
            if (u1,events) in rm:
                u1 = rm[(u1,events)]
        else:
            print("Forbidden action")
    game.show_map()
    print("\nGame over!")
    print("reward:", reward)
    print("time:", t)
    input()

    return reward, len(trace), trace
