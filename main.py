#pylint:disable=C0103
'''Poker simulation environment
Controls:
Make 4*current best model (random if first episode)
Start episode
For game in episode:
    Reset game to clean slate
    For round in game:
        For step in round:
            Advance game state using action provided by the model whose turn it is, given game state at step
            If not first step in game:
                Add (prev_state, prev_action, prev_reward, current_state, done) to memory of current player
            prev_state = current_state
            prev_action = action
            prev_reward = current_reward
            Progress to next player's step
    If game index == batch_size for memory replay:
        Do a memory replay for every player using their memories
End episode
Pick best model
Lather, rinse, repeat for n episodes
'''
from agent.dqn_agent import DQNAgent
from environment.sample_state import SAMPLE_STATE, SAMPLE_HOLE_CARDS, SAMPLE_ACTIONS
from environment.dqn_agent_wrapper import DQNAgentWrapper
from pypokerengine.api.emulator import Emulator

N_AGENTS = 4
BB_SIZE = 10
STACK_SIZE = 200
N_EPISODES = 1
GAMES_PER_EPISODE = 1
REPLAY_EVERY_N_GAMES = 32
BATCH_SIZE = REPLAY_EVERY_N_GAMES
N_ACTIONS = 7

emulator = Emulator()

# used only for calculating # of features
_sample_features = DQNAgent(3, 3).make_features(SAMPLE_ACTIONS, SAMPLE_HOLE_CARDS, SAMPLE_STATE)
STATE_SIZE = len(_sample_features)

agents = [DQNAgent(STATE_SIZE, N_ACTIONS)] * N_AGENTS

for e in range(N_EPISODES):
    temp_final_state = {}
    for game in range(GAMES_PER_EPISODE):
        wrappers = []
        player_info = {}
        for i, agent in enumerate(agents):
            if temp_final_state:
                for seat in temp_final_state['seats']:
                    if seat['uuid'] == i:
                        player_info[i] = {'name': 'Player ' + i, 'stack': seat['stack']}
                        break
            else:
                player_info[i] = {'name': 'Player ' + i, 'stack': STACK_SIZE}
            wrappers.append(DQNAgentWrapper(agent))
            emulator.register_player(uuid=i, player=wrappers[-1])

        initial_game_state = emulator.generate_initial_game_state(player_info)
        game_finish_state, events = emulator.run_until_game_finish(initial_game_state)

        # state = env.reset()
        # done = False
        # while not done:
        #     current_player_idx = state['current_player_idx']
        #     current_agent = agents[current_player_idx]
        #     action = current_agent['agent'].act(state)
        #     next_state, reward, done = env.step(action)
        #     if current_agent['prev_state'] is not None: # not first step
        #     agents[current_player_idx] = current_agent
        #     state = next_state
        agents[:] = [wrapper.agent for wrapper in wrappers]
        temp_final_state = wrappers[0].final_state

        if game % REPLAY_EVERY_N_GAMES == 0 or game == GAMES_PER_EPISODE-1:
            # replay memory for every agent
            map(lambda a: a.replay(BATCH_SIZE), agents)
        print('Episode {} over'.format(e))

    # Pick best model
    highest_roll, highest_idx = 0, None
    for seat in temp_final_state['seats']:
        if seat['stack'] > highest_roll:
            highest_roll = seat['stack']
            highest_idx = seat['uuid']
    agents[highest_idx].reset_states()  # clear memory of RNN
    agents = [agents[highest_idx]] * N_AGENTS
