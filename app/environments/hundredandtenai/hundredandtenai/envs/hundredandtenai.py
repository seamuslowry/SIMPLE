
from enum import IntEnum
from gym import Env, spaces
import numpy as np

from hundredandten import HundredAndTen, constants, GameStatus, deck, RoundStatus, BidAmount, Play, Bid, Discard, SelectTrump, SelectableSuit

from stable_baselines import logger

# static actions must be numerically above card indices
# selecting a number that corresponds to a card in the deck
# indicates an attempt to play that card
class StaticActions(IntEnum):
    PASS = len(deck.cards)
    FIFTEEN = len(deck.cards) + 1
    TWENTY = len(deck.cards) + 2
    TWENTY_FIVE = len(deck.cards) + 3
    THIRTY = len(deck.cards) + 4
    SHOOT_THE_MOON = len(deck.cards) + 5
    SELECT_CLUBS = len(deck.cards) + 6
    SELECT_SPADES = len(deck.cards) + 7
    SELECT_DIAMONDS = len(deck.cards) + 8
    SELECT_HEARTS = len(deck.cards) + 9
    DISCARD = len(deck.cards) + 10

TOTAL_AVAILABLE_ACTIONS = len(deck.cards) + len(StaticActions)

TOTAL_OBSERVATIONS = (
    # cards in play (1)
    # cards in hand (.5)
    # cards unknown (0)
    # cards played (-.5)
    # cards discarded (-1)
    len(deck.cards) + 
    # current bid amount (as percentage of max: 60)
    1 + 
    # which player is bidding (relative to them, normalize between [0,1])
    1 + 
    # all player's current trick scores (as a percentage of max: 30)
    4 + 
    # all player's current scores (as a percentage of max: 165)
    4 +
    # what suit is trump (0, .25, .50, .75)
    1 +
    # all legal actions
    TOTAL_AVAILABLE_ACTIONS)

MAX_BID = 60
MAX_TRICK_SCORE = 30

class HundredAndTenEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(HundredAndTenEnv, self).__init__()
        self.name = 'hundredandtenai'
        self.manual = manual

        self.game = HundredAndTen()
        self.game.join('0')
        self.game.join('1')
        self.game.join('2')
        self.game.join('3')
        self.game.start_game()
        
        self.n_players = 4
        self.current_player_num = int(self.game.active_round.active_player.identifier)

        self.action_space = spaces.Discrete(TOTAL_AVAILABLE_ACTIONS)
        self.observation_space = spaces.Box(-1, 1, (TOTAL_OBSERVATIONS,))
        self.verbose = verbose
        self.done = False

        
    @property
    def observation(self):
        obs = np.zeros(len(deck.cards))
        active_bidder = self.game.active_round.active_bidder

        # card observations
        for card_index in range(len(deck.cards)):
            card = deck.cards[card_index]
            if self.done:
                continue

            # already played is a -.5
            if card in list(map(lambda p: p.card, [play for plays in map(lambda t: t.plays, self.game.active_round.tricks) for play in plays])):
                obs[card_index] = -.5
            # in play is a 1 (do this after already played to overwrite)
            if self.game.active_round.tricks and card in list(map(lambda p: p.card,self.game.active_round.active_trick.plays)):
                obs[card_index] = 1
            # in the player's hand is a .5
            if card in self.game.active_round.active_player.hand:
                obs[card_index] = .5
            # discarded is -1
            if card in ([discard for discard in self.game.active_round.discards if discard.identifier == self.game.active_round.active_player.identifier] or [Discard('', [])])[0].cards:
                obs[card_index] = -1

        # current bid amount observation
        obs = np.append(obs, (self.game.active_round.active_bid or 0) / MAX_BID)

        # current bidding player observation
        obs = np.append(obs, ((self.current_player_num - int(active_bidder.identifier)) % 2) / 2 if active_bidder else -1)

        # current trick scores observation
        scores = self.game.active_round.scores
        obs = np.append(obs, sum(map(lambda s: s.value, [score for score in scores if score.identifier == '0'])) / MAX_TRICK_SCORE)
        obs = np.append(obs, sum(map(lambda s: s.value, [score for score in scores if score.identifier == '1'])) / MAX_TRICK_SCORE)
        obs = np.append(obs, sum(map(lambda s: s.value, [score for score in scores if score.identifier == '2'])) / MAX_TRICK_SCORE)
        obs = np.append(obs, sum(map(lambda s: s.value, [score for score in scores if score.identifier == '3'])) / MAX_TRICK_SCORE)

        # current game scores observation
        game_scores = self.game.scores
        obs = np.append(obs, game_scores['0'] / constants.WINNING_SCORE)
        obs = np.append(obs, game_scores['1'] / constants.WINNING_SCORE)
        obs = np.append(obs, game_scores['2'] / constants.WINNING_SCORE)
        obs = np.append(obs, game_scores['3'] / constants.WINNING_SCORE)

        # trump observation
        trump = self.game.active_round.trump
        obs = np.append(obs, trump.value / 4 if trump else -1)

        # all legal actions observation
        obs = np.append(obs, self.legal_actions)

        return obs

    @property
    def legal_actions(self):
        legal_actions = np.zeros(TOTAL_AVAILABLE_ACTIONS)
        status = self.game.status

        if self.done:
            return legal_actions

        active_player_trump_cards = [
            card for card in self.game.active_round.active_player.hand
            if card.suit == self.game.active_round.trump or card.always_trump]

        # check if playing each card is legal
        for card_index in range(len(deck.cards)):
            card = deck.cards[card_index]
            # playing a card is only legal in the trick stage and for cards the player has in hand
            if status == RoundStatus.TRICKS and card in self.game.active_round.active_player.hand:
                # additionally, if the trick is bleeding, the card must be trump or the player must have no trumps
                if not self.game.active_round.active_trick.bleeding or not active_player_trump_cards or card in active_player_trump_cards:
                    legal_actions[card_index] = 1

        # check if all the static actions are legal

        # bidding actions are only legal during the bidding stage
        # and if the current player has the bid amount in their available bids
        bidding = status == RoundStatus.BIDDING
        available_bids = self.game.active_round.available_bids(str(self.current_player_num))

        legal_actions[StaticActions.PASS.value] = bidding and BidAmount.PASS in available_bids
        legal_actions[StaticActions.FIFTEEN.value] = bidding and BidAmount.FIFTEEN in available_bids
        legal_actions[StaticActions.TWENTY.value] = bidding and BidAmount.TWENTY in available_bids and self.game.active_round.active_bid == BidAmount.FIFTEEN
        legal_actions[StaticActions.TWENTY_FIVE.value] = bidding and BidAmount.TWENTY_FIVE in available_bids and self.game.active_round.active_bid == BidAmount.TWENTY
        legal_actions[StaticActions.THIRTY.value] = bidding and BidAmount.THIRTY in available_bids and self.game.active_round.active_bid == BidAmount.TWENTY_FIVE
        legal_actions[StaticActions.SHOOT_THE_MOON.value] = 0 # bidding and BidAmount.SHOOT_THE_MOON in available_bids
        
        # select trump actions are only available if the stage is trump selection
        selecting_trump = status == RoundStatus.TRUMP_SELECTION
        legal_actions[StaticActions.SELECT_CLUBS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_DIAMONDS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_HEARTS.value] = selecting_trump
        legal_actions[StaticActions.SELECT_SPADES.value] = selecting_trump

        # discards action is only available if the stage is discarding
        legal_actions[StaticActions.DISCARD.value] = status == RoundStatus.DISCARD
        
        return legal_actions.astype(int)


    def step(self, action):

        reward = [0] * self.n_players

        if not self.legal_actions[action]:
            reward = [1.0/(self.n_players-1)] * self.n_players
            reward[self.current_player_num] = -1
            return self.observation, reward, True, {}
        
        # play a card action
        if action < len(deck.cards):  # type: ignore action will be comparable
            self.game.act(Play(str(self.current_player_num), deck.cards[action])) # type: ignore action will be comparable
        
        # bid action
        if action == StaticActions.PASS:
            self.game.act(Bid(str(self.current_player_num), BidAmount.PASS))
        if action == StaticActions.FIFTEEN:
            self.game.act(Bid(str(self.current_player_num), BidAmount.FIFTEEN))
        if action == StaticActions.TWENTY:
            self.game.act(Bid(str(self.current_player_num), BidAmount.TWENTY))
        if action == StaticActions.TWENTY_FIVE:
            self.game.act(Bid(str(self.current_player_num), BidAmount.TWENTY_FIVE))
        if action == StaticActions.THIRTY:
            self.game.act(Bid(str(self.current_player_num), BidAmount.THIRTY))
        if action == StaticActions.SHOOT_THE_MOON:
            self.game.act(Bid(str(self.current_player_num), BidAmount.SHOOT_THE_MOON))

        # select trump action
        if action == StaticActions.SELECT_CLUBS:
            self.game.act(SelectTrump(str(self.current_player_num), SelectableSuit.CLUBS))
        if action == StaticActions.SELECT_DIAMONDS:
            self.game.act(SelectTrump(str(self.current_player_num), SelectableSuit.DIAMONDS))
        if action == StaticActions.SELECT_HEARTS:
            self.game.act(SelectTrump(str(self.current_player_num), SelectableSuit.HEARTS))
        if action == StaticActions.SELECT_SPADES:
            self.game.act(SelectTrump(str(self.current_player_num), SelectableSuit.SPADES))

        # discard action
        if action == StaticActions.DISCARD:
            self.game.act(Discard(str(self.current_player_num), [card for card in self.game.active_round.active_player.hand if (card.suit != self.game.active_round.trump or not card.always_trump)]))

        winner = self.game.winner
        self.done = bool(winner)
        self.current_player_num = int(self.game.active_round.active_player.identifier) if not self.done else 0

        scores = self.game.scores

        reward[0] = scores['0']
        reward[1] = scores['1']
        reward[2] = scores['2']
        reward[3] = scores['3']

        if winner:
            reward[int(winner.identifier)] += 60

        return self.observation, reward, self.done, {}

    def reset(self):
        # reset game

        self.game = HundredAndTen()
        self.game.join('0')
        self.game.join('1')
        self.game.join('2')
        self.game.join('3')
        self.game.start_game()

        self.current_player_num = int(self.game.active_round.active_player.identifier)

        self.done = False

        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        
        if close:
            return

        status = self.game.status
        active_bid = self.game.active_round.active_bid
        active_bidder = self.game.active_round.active_bidder

        logger.debug(f'\n\n-------STATUS {status}-----------')
        logger.debug(f"It is Player {self.current_player_num}'s turn to play")

        if status != GameStatus.WON:
            logger.debug(f'Active Bid: {active_bid if active_bid else "N/A"}')
            logger.debug(f'Active Bidder: {active_bidder.identifier if active_bidder else "N/A"}')
            logger.debug(f'Trump: {self.game.active_round.trump if self.game.active_round.trump else "N/A"}')



            if (self.game.status == RoundStatus.TRICKS):
                logger.debug(f'Played cards: {self.game.active_round.active_trick.plays}')
                logger.debug(f'Bleeding: {self.game.active_round.active_trick.bleeding}')



            logger.debug(f'Player {self.current_player_num}\'s hand')
            for card in self.game.active_round.active_player.hand:
                logger.debug(f'{card.number.name} of {card.suit.name} (#{deck.cards.index(card)})')

            logger.debug(f'Current scores: {self.game.scores}')
        else:
            logger.debug(f'Winner: {self.game.winner}')
            logger.debug(f'Final Scores: {self.game.scores}')
        
        if not self.done:
            logger.debug(f'Legal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Sushi Go!')
