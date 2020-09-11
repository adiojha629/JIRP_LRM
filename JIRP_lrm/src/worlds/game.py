from worlds.traffic_world import TrafficWorldParams, TrafficWorld
from worlds.craft_world import CraftWorldParams, CraftWorld
from worlds.office_world import OfficeWorldParams, OfficeWorld,OfficeWorldActive

class GameParams:
    """
    Auxiliary class with the configuration parameters that the Game class needs
    """
    def __init__(self, game_params):
        self.game_type   = game_params.game_type
        self.game_params = game_params
        if self.game_type not in ["craftworld", "trafficworld", "officeworld"]:
            print(self.game_type, "is not currently supported")
            exit()

class Game:
#note that not all methods are defined in officeworld; some methods may not be needed. they are included here for the sake of being consistant with rodrigo's code
    def __init__(self, params,label):
        self.params = params
        self.restart(label) #label used for debugging which environment is used for testing and training in LRM 9.5.2020

    def restart(self,label):
        #print("Label is ",label)
        if self.params.game_type == "craftworld":
            self.game = CraftWorld(self.params)
        if self.params.game_type == "trafficworld":
            self.game = TrafficWorld(self.params.game_params)
        if self.params.game_type == "officeworld":
            self.game = OfficeWorld(self.params)
        if self.params.game_type == "officeworld_active":
            self.game = OfficeWorldActive(self.params)

    def get_game(self):
        return self.game
    def is_env_game_over(self):
        return self.game.env_game_over
    def get_true_propositions(self):
        return self.game.get_events()
    def execute_action(self, action):
        """
        We execute 'action' in the game
        Returns the reward
        """
        return self.game.execute_action(action)
    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.game.get_actions()

    def get_last_action(self):
        """
        Returns agent's last performed action
        """
        return self.game.get_last_action()

    def get_true_propositions_action(self,a):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.game.get_true_propositions_action(a)

    def get_state(self):
        """
        Returns a representation of the current state with enough information to 
        compute a reward function using an RM (the format is domain specific)
        """
        return self.game.get_state()

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        return self.game.get_features()
    def get_optimal_action(self):
        """
        HACK: returns the best possible action given current state
        """
        return self.game.get_optimal_action()
    def get_state_and_features(self):
        return self.get_state(), self.get_features()
    def get_perfect_rm(self):
        """
        Returns a perfect RM for this domain
        """
        return self.game.get_perfect_rm()
    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        return self.game.get_events()
    def get_is_done(self):
        return self.game.get_is_done()
    def get_all_events(self):
        """
        Returns a string with all the possible events that may occur in the environment
        """
        return self.game.get_all_events()
    def get_location(self):
        # this auxiliary method allows to keep track of the agent's movements
        return self.game.get_location()
