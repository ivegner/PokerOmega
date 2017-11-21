from pypokerengine.players import BasePokerPlayer
import numpy as np

RAISE_AMTS = [3, 6, 9, 12, 15, 18]
class DQNAgentWrapper(BasePokerPlayer):
    def __init__(self, agent, init_stack_size):
        super(DQNAgentWrapper, self).__init__()
        self.agent = agent
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.player_idx = None
        self.player_uuid = None
        self.bb_amount = None
        self.init_stack_size = init_stack_size
        self.final_state = None

    def declare_action(self, valid_actions, hole_cards, game_state):
        if self.player_idx is None:
            self.player_idx = game_state['next_player']
            self.player_uuid = game_state['seats'][self.player_idx]['uuid']
            self.bb_amount = game_state['small_blind_amount'] * 2

        features = self.agent.make_features(valid_actions, hole_cards, game_state)
        actions = self.agent.act(features)
        action_str, chosen_action, amount = None, None, 0
        sorted_by_best_action = np.argsort(actions)[::-1]   # first entry is index of best action

        _valid_action, _valid_action_idx = False, 0
        while not _valid_action:     # find best valid action
            chosen_action = sorted_by_best_action[_valid_action_idx]
            if chosen_action == 0:
                action_str = 'fold'
                amount = 0
                _valid_action = True
                break
            elif chosen_action == 1:
                action_str = 'call'
                amount = valid_actions[1]['amount']
                _valid_action = True
                break
            else:
                action_str = 'raise'
                amount = RAISE_AMTS[chosen_action-2] * self.bb_amount
                if valid_actions[2]['amount']['min'] <= amount <= valid_actions[2]['amount']['max']:
                    _valid_action = True
                    break
                else:
                    _valid_action_idx += 1

        if self.prev_state is not None:
            self.agent.remember(self.prev_state, self.prev_action, self.prev_reward, features, 0)
        self.prev_state = features
        self.prev_action = chosen_action
        self.prev_reward = 0
        return action_str, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass
