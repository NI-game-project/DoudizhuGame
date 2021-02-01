import numpy as np
from envs.env import Env


class DoudizhuEnv(Env):
    '''
    Doudizhu Environment
    '''

    def __init__(self, config):
        from doudizhu.utils import SPECIFIC_MAP, CARD_RANK_STR
        from doudizhu.utils import ACTION_LIST, ACTION_SPACE
        from doudizhu.utils import encode_cards
        from doudizhu.utils import cards2str, cards2str_with_suit
        from doudizhu.game import DoudizhuGame as Game
        self._encode_cards = encode_cards
        self._cards2str = cards2str
        self._cards2str_with_suit = cards2str_with_suit
        self._SPECIFIC_MAP = SPECIFIC_MAP
        self._CARD_RANK_STR = CARD_RANK_STR
        self._ACTION_LIST = ACTION_LIST
        self._ACTION_SPACE = ACTION_SPACE

        self.name = 'doudizhu'
        self.game = Game()
        super().__init__(config)
        self.state_shape = [8, 5, 15]

    def _extract_state(self, state):
        ###### only changed this function and added get_hand_length func in doudizhu.py
        ###### the rest is the same
        ''' Encode state
        Args:
            state (dict): dict of original state
        Returns:
            numpy array: 8*5*15 array
                         8 : 1. current hand
                             2. the union of the other two players' hand
                             3,4,5. the recent three actions taken by the current player
                             6-7: representing player's role
                                6. 1s if landlord, 0s otherwise
                                7. 0s for peasant1, 1s for peasant2
                             8.  1-3 row: current hand_length of all players, in the order(landlord, peasant1, peasant2)

        '''

        obs = np.zeros((8, 5, 15), dtype=int)
        ll_hand_length, p1_hand_length, p2_hand_length = self.get_hand_length(state)
        # calculate the length of the each player's hand using 'trace' of the state_obs
        for index in range(5):
            obs[index][0] = np.ones(15, dtype=int)
        self._encode_cards(obs[0], state['current_hand'])
        self._encode_cards(obs[1], state['others_hand'])
        for i, action in enumerate(state['trace'][-9::3]):
            if action[1] != 'pass':
                self._encode_cards(obs[4 - i], action[1])
        if state['self'] == 0:
            obs[5][:] = np.ones(15, dtype=int)
        elif state['self'] == 1:
            obs[6][:] = np.zeros(15, dtype=int)
        elif state['self'] == 2:
            obs[6][:] = np.ones(15, dtype=int)
        for index in range(3):
            obs[7][index] = np.zeros(15, dtype=int)
        for i in range(ll_hand_length):
            obs[7][0][i] = 1
        for i in range(p1_hand_length):
            obs[7][1][i] = 1
        for i in range(p2_hand_length):
            obs[7][2][i] = 1

        extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions()}
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            # TODO: state['actions'] can be None, may have bugs
            if state['actions'] == None:
                extracted_state['raw_legal_actions'] = []
            else:
                extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        '''
        Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def get_winner_id(self):
        return self.game.get_winner_id()

    def _decode_action(self, action_id):
        '''
        Action id -> the action in the doudizhu. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the doudizhu engine.
        '''
        abstract_action = self._ACTION_LIST[action_id]
        # without kicker
        if '*' not in abstract_action:
            return abstract_action
        # with kicker
        legal_actions = self.game.state['actions']
        specific_actions = []
        kickers = []
        for legal_action in legal_actions:
            for abstract in self._SPECIFIC_MAP[legal_action]:
                main = abstract.strip('*')
                if abstract == abstract_action:
                    specific_actions.append(legal_action)
                    kickers.append(legal_action.replace(main, '', 1))
                    break
        # choose kicker with minimum score
        player_id = self.game.get_player_id()
        kicker_scores = []
        for kicker in kickers:
            score = 0
            for action in self.game.judger.playable_cards[player_id]:
                if kicker in action:
                    score += 1
            kicker_scores.append(score + self._CARD_RANK_STR.index(kicker[0]))
        min_index = 0
        min_score = kicker_scores[0]
        for index, score in enumerate(kicker_scores):
            if score < min_score:
                min_score = score
                min_index = index
        return specific_actions[min_index]

    def _get_legal_actions(self):
        '''
        Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_action_id = []
        legal_actions = self.game.state['actions']
        if legal_actions:
            for action in legal_actions:
                for abstract in self._SPECIFIC_MAP[action]:
                    action_id = self._ACTION_SPACE[abstract]
                    if action_id not in legal_action_id:
                        legal_action_id.append(action_id)
        return legal_action_id

    def get_perfect_information(self):
        '''
        Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['hand_cards_with_suit'] = [self._cards2str_with_suit(player.current_hand) for player in self.game.players]
        state['hand_cards'] = [self._cards2str(player.current_hand) for player in self.game.players]
        state['landlord'] = self.game.state['landlord']
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state

    def get_hand_length(self, state):
        # calculate the length of each player's current hand using trace
        # just for encoding the state, kind of a naive approach,
        # but we don't need to mess up with the state of different player or env.py

        landlord_hand_length = 0
        p1_hand_length = 0
        p2_hand_length = 0
        for action in (state['trace']):
            if action[0] == 0:
                if action[1] != 'pass':
                    landlord_hand_length += len(action[1])
            elif action[0] == 1:
                if action[1] != 'pass':
                    p1_hand_length += len(action[1])
            elif action[0] == 2:
                if action[1] != 'pass':
                    p2_hand_length += len(action[1])

        # if the hand length is larger than 15, set it to 15 to match the shape of the state, since 15 or larger than
        # 15 won't really make a difference.
        landlord_hand_length = min((20 - landlord_hand_length), 15)
        p1_hand_length = min((17 - p1_hand_length), 15)
        p2_hand_length = min((17 - p2_hand_length), 15)
        return landlord_hand_length, p1_hand_length, p2_hand_length




