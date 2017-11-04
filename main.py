# pylint:disable=C0103
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
import numpy as np

N_AGENTS = 4
BB_SIZE = 10
STACK_SIZE = 200
N_EPISODES = 40
GAMES_PER_EPISODE = 100
REPLAY_EVERY_N_GAMES = 32
BATCH_SIZE = REPLAY_EVERY_N_GAMES
N_ACTIONS = 7
EVAL_EVERY_N_EPISODES = 5
USE_ROLL_INSTEAD_OF_WIN_COUNT = False

def run_episode(agents):
    emulator = Emulator()
    temp_final_state = {}
    winner_counts = [0] * N_AGENTS
    n_games_played = 0
    for game in range(GAMES_PER_EPISODE):
        wrappers = []
        player_info = {}
        for i, agent in enumerate(agents):
            # if temp_final_state:
            #     for seat in temp_final_state:
            #         player_info[seat.uuid] = {'name': 'Player ' + str(seat.uuid), 'stack': seat.stack if seat.stack else STACK_SIZE}
            # else:
            player_info[i] = {'name': 'Player ' + str(i), 'stack': STACK_SIZE}
            wrappers.append(DQNAgentWrapper(agent))
            emulator.register_player(uuid=i, player=wrappers[-1])
        emulator.set_game_rule(N_AGENTS, 2, BB_SIZE / 2, 0)
        initial_game_state = emulator.generate_initial_game_state(player_info)

        game_state, events = emulator.start_new_round(initial_game_state)
        game_finish_state, events = emulator.run_until_game_finish(game_state)

        # if len(events) > 1000:
        #     print('dumping')
        #     with open('event_dump_' + str(game), 'w') as f:
        #         json.dump(events, f, indent=2)

        winner = events[-2]['winners'][0]['uuid']
        winner_counts[winner] += 1
        n_games_played += 1

        agents[:] = [wrapper.agent for wrapper in wrappers]
        temp_final_state = game_finish_state['table'].seats.players

        # print('====')
        print('\rGame {} out of {}'.format(game, GAMES_PER_EPISODE), end='')
        # print(game_finish_state)
        # print('\n')
        # print(events[-5:])
        # print('====')
        if game % REPLAY_EVERY_N_GAMES == 0 or game == GAMES_PER_EPISODE - 1:
            # replay memory for every agent
            map(lambda a: a.replay(BATCH_SIZE), agents)
    return agents, temp_final_state, winner_counts, n_games_played

if __name__ == '__main__':
    # used only for calculating # of features
    _sample_features = DQNAgent(3, 3).make_features(SAMPLE_ACTIONS, SAMPLE_HOLE_CARDS, SAMPLE_STATE)
    STATE_SIZE = len(_sample_features)

    oldest_agents = [DQNAgent(STATE_SIZE, N_ACTIONS)] * N_AGENTS
    old_agents = [DQNAgent(STATE_SIZE, N_ACTIONS)] * N_AGENTS
    agents = [DQNAgent(STATE_SIZE, N_ACTIONS)] * N_AGENTS

    for e in range(N_EPISODES):
        new_agents, final_state, winner_counts, n_games_played = run_episode(agents)
        print('\nEpisode {} over'.format(e))
        # Pick best model
        highest_idx = None
        if USE_ROLL_INSTEAD_OF_WIN_COUNT:
            highest_roll = 0
            for seat in final_state:
                if seat.stack > highest_roll:
                    highest_roll = seat.stack
                    highest_idx = seat.uuid
        else:
            highest_idx = np.argmax(winner_counts)
        new_agents[highest_idx].model.reset_states()  # clear memory of RNN

        if e == N_EPISODES-1 or e % EVAL_EVERY_N_EPISODES == 0:
            print('====')
            print('Final evaluation')
            _, final_state, winner_counts, n_games_played = run_episode([agents[0]] + oldest_agents[:-1])
            print('\nNewest best won against oldest {} percent of games'.format((winner_counts[0] / n_games_played) * 100))
            print('====')

        # elif e % EVAL_EVERY_N_EPISODES == 0: # run 3x old versions against 1 new version
        #     print('====')
        #     print('Evaluating')
        #     _, final_state, winner_counts, n_games_played = run_episode([agents[0]] + old_agents[:-1])
        #     print('\nNew won against old {} percent of games'.format((winner_counts[0] / n_games_played) * 100))
        #     print('====')
        #     old_agents = agents

        agents = [new_agents[highest_idx]] * N_AGENTS


    agents[0].save('pokerai_rl.h5')