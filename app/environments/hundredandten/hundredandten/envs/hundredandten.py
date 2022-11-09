
from enum import IntEnum
from gym import Env, spaces
import numpy as np

from hundredandten import HundredAndTen, deck, RoundStatus, BidAmount

from stable_baselines import logger

# static actions must be numerically above card indices
# selecting a number that corresponds to a card in the deck
# indicates an attempt to play that card
class StaticActions(IntEnum):
    PASS = len(deck.cards) + 1
    FIFTEEN = len(deck.cards) + 2
    TWENTY = len(deck.cards) + 3
    TWENTY_FIVE = len(deck.cards) + 4
    THIRTY = len(deck.cards) + 5
    SHOOT_THE_MOON = len(deck.cards) + 6
    SELECT_CLUBS = len(deck.cards) + 7
    SELECT_SPADES = len(deck.cards) + 8
    SELECT_DIAMONDS = len(deck.cards) + 9
    SELECT_HEARTS = len(deck.cards) + 10
    DISCARD = len(deck.cards) + 11

TOTAL_AVAILABLE_ACTIONS = len(deck.cards) + len(StaticActions)


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

        self.action_space = spaces.Discrete(TOTAL_AVAILABLE_ACTIONS)
        self.observation_space = spaces.Box(0, 1, (0,)) # TODO need to describe observation space
        self.verbose = verbose

        
    @property
    def observation(self):
        obs = np.zeros(([5, 53]))

        ret = obs.flatten()

        return ret

    @property
    def legal_actions(self):
        legal_actions = np.zeros(TOTAL_AVAILABLE_ACTIONS)
        status = self.game.status

        # check if playing each card is legal
        for card_index in range(len(deck.cards)):
            card = deck.cards[card_index]
            # playing a card is only legal in the trick stage and for cards the player has in hand
            if status == RoundStatus.TRICKS and card in self.game.active_round.active_player.hand:
                # additionally, if the trick is bleeding, the card must NOT be bleeding OR the card must be trump
                if not self.game.active_round.active_trick.bleeding or (card.suit == self.game.active_round.trump or card.always_trump):
                    legal_actions[card_index] = 1

        # check if all the static actions are legal

        # bidding actions are only legal during the bidding stage
        # and if the current player has the bid amount in their available bids
        bidding = status == RoundStatus.BIDDING
        available_bids = self.game.active_round.available_bids(str(self.current_player_num))

        legal_actions[StaticActions.PASS.value] = bidding and BidAmount.PASS in available_bids
        legal_actions[StaticActions.FIFTEEN.value] = bidding and BidAmount.FIFTEEN in available_bids
        legal_actions[StaticActions.TWENTY.value] = bidding and BidAmount.TWENTY in available_bids
        legal_actions[StaticActions.TWENTY_FIVE.value] = bidding and BidAmount.TWENTY_FIVE in available_bids
        legal_actions[StaticActions.THIRTY.value] = bidding and BidAmount.THIRTY in available_bids
        legal_actions[StaticActions.SHOOT_THE_MOON.value] = bidding and BidAmount.SHOOT_THE_MOON in available_bids
        
        # select trump actions are only available if the stage is trump selection
        selecting_trump = status == RoundStatus.TRUMP_SELECTION
        legal_actions[StaticActions.SELECT_CLUBS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_DIAMONDS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_HEARTS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_SPADES.value] = selecting_trump

        # discards action is only available if the stage is discarding
        selecting_trump = status == RoundStatus.TRUMP_SELECTION
        legal_actions[StaticActions.DISCARD.value] = status == RoundStatus.DISCARD
        
        return legal_actions.astype(int)


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
