import random
from collections import deque

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = 132
        self.action_size = action_size
        self.memory = deque()
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.num_agents = num_agents

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(SimpleRNN(64, input_shape=(1, self.state_size), activation='relu', return_sequences=True))
        model.add(SimpleRNN(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = state.reshape((1, 1, len(state)))
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([state])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1,1,len(state)))
            next_state = next_state.reshape((1,1,len(next_state)))
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def _explode(self, array):
        return [[a] for a in array]

    def make_features(self, valid_actions, hole_cards, game_state):
        player_idx = game_state['next_player']
        player_uuid = game_state['seats'][player_idx]['uuid']
        bb_amount = game_state['small_blind_amount'] * 2

        # split hole cards, onehot suits and values
        suit_to_int_enc = LabelEncoder().fit(['H', 'S', 'D', 'C'])
        suit_int_to_onehot_enc = OneHotEncoder(sparse=False).fit(self._explode(range(0, 4)))
        value_int_to_onehot_enc = OneHotEncoder(sparse=False).fit(self._explode(range(2, 15)))

        hole_suits, hole_values = self._cards_to_arrays(hole_cards)
        hole_suits = suit_to_int_enc.transform(self._explode(hole_suits))
        hole_suits = suit_int_to_onehot_enc.transform(self._explode(hole_suits))
        hole_values = value_int_to_onehot_enc.transform(self._explode(hole_values))

        # river cards
        temp_suit_zeros = np.zeros((5, 4))
        temp_value_zeros = np.zeros((5, 13))
        river_suits, river_values = self._cards_to_arrays(game_state['community_card'])
        if river_suits and river_values:
            river_suits = suit_to_int_enc.transform(self._explode(river_suits))
            river_suits = suit_int_to_onehot_enc.transform(self._explode(river_suits))
            river_values = value_int_to_onehot_enc.transform(self._explode(river_values))
            temp_suit_zeros[:river_suits.shape[0], :river_suits.shape[1]] = river_suits
            temp_value_zeros[:river_values.shape[0], :river_values.shape[1]] = river_values

        river_suits = temp_suit_zeros
        river_values = temp_value_zeros

        # pot
        total_main_amount = game_state['pot']['main']['amount']
        total_side_pot = sum([a['amount'] for a in game_state['pot']['side']])
        total_pot_as_bb = [(total_main_amount + total_side_pot) / bb_amount]

        # own stack size
        own_stack_size = [game_state['seats'][player_idx]['stack'] / bb_amount]

        # other players stack size
        players_after_stacks = [p['stack'] / bb_amount for p in game_state['seats'][player_idx + 1:]]
        players_before_stacks = [p['stack'] / bb_amount for p in game_state['seats'][:player_idx]]
        players_after_stacks.extend(players_before_stacks)
        other_players_stack_sizes = players_after_stacks

        # TODO: distance from button, scaled from 0 to 1
        # n_players = len(game_state['seats'])
        # distance = ()

        # players folded? (binary)
        players_after_folds = [int(p['state'] == 'folded') for p in game_state['seats'][player_idx + 1:]]
        players_before_folds = [int(p['state'] == 'folded') for p in game_state['seats'][:player_idx]]
        players_after_folds.extend(players_before_folds)
        player_folds = players_after_folds

        # action history, for use below
        game_state_histories = (game_state['action_histories'].values())
        action_history = [action for phase in game_state_histories for action in phase]

        # money put into pot by each player since our last
        moves_since_our_last = []
        for action in action_history[::-1]:
            if action['uuid'] != player_uuid:
                moves_since_our_last.append(action)
            else:
                break
        moves_since_our_last.reverse()
        moves_since_our_last = moves_since_our_last[:self.num_agents]
        temp_move_zeroes = np.zeros(self.num_agents)
        money_since_our_last_move = [a.get('amount', 0) for a in moves_since_our_last]
        for i, m in enumerate(money_since_our_last_move):
            temp_move_zeroes[i] = m
        money_since_our_last_move = temp_move_zeroes


        # amt to call
        amt_to_call = [0]
        for action in valid_actions:
            if action['action'] == 'call':
                amt_to_call = [action['amount'] / bb_amount]
                break

        feature_arrays = [hole_values, hole_suits, river_values, river_suits, total_pot_as_bb,
                own_stack_size, other_players_stack_sizes, player_folds, money_since_our_last_move,
                amt_to_call]

        ret = None

        for array in feature_arrays:
            array = np.array(array).flatten()
            if ret is not None:
                ret = np.concatenate((ret, array))
            else:
                ret = array

        return ret

    def _cards_to_arrays(self, cards):
        suits = []
        values = []
        for card in cards:
            if card[1:] == 'A': card = card[0] + '14'
            if card[1:] == 'K': card = card[0] + '13'
            if card[1:] == 'Q': card = card[0] + '12'
            if card[1:] == 'J': card = card[0] + '11'
            if card[1:] == 'T': card = card[0] + '10'
            suits.append(card[0])
            values.append(int(card[1:]))
        return suits, values
