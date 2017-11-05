from pypokerengine.players import BasePokerPlayer

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
        action_idx = self.agent.act(features)
        action, amount = None, 0
        if action_idx == 0:
            action = 'fold'
        if action_idx == 1:
            action = 'call'
            amount = valid_actions[1]['amount']
        else:
            action = 'raise'
            amount = RAISE_AMTS[action_idx-2] * self.bb_amount

        if self.prev_state is not None:
            self.agent.remember(self.prev_state, self.prev_action, self.prev_reward, features, 0)
        self.prev_state = features
        self.prev_action = action_idx
        self.prev_reward = 0
        return action, amount

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