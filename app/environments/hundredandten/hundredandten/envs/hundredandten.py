
from enum import IntEnum
from gym import Env, spaces
import numpy as np

from hundredandten import HundredAndTen

from stable_baselines import logger

class Actions(IntEnum):
    PASS = 0

class HundredAndTenEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(HundredAndTenEnv, self).__init__()
        self.name = 'hundredandten'
        self.manual = manual

        self.game = HundredAndTen()
        self.game.join('1')
        self.game.join('2')
        self.game.join('3')
        self.game.join('4')
        
        self.n_players = 4
        self.current_player_num = 0

        self.action_space = spaces.Discrete(0) # TODO need to describe action space
        self.observation_space = spaces.Box(0, 1, (0,)) # TODO need to describe observation space
        self.verbose = verbose

        
    @property
    def observation(self):
        obs = np.zeros(([5, 53]))

        ret = obs.flatten()

        return ret

    @property
    def legal_actions(self):
        legal_actions = np.zeros(0) # TODO match action space size
        
        return legal_actions


    def step(self, action):
        
        # TODO process action

        return self.observation, [], False, {}

    def reset(self):
        # reset game

        self.game = HundredAndTen()

        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        
        if close:
            return

    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Sushi Go!')
