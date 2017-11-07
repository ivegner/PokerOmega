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
import sys
import copy

np.random.seed(12)

N_AGENTS = 4
BB_SIZE = 10
STACK_SIZE = 200
N_EPISODES = 2000
GAMES_PER_EPISODE = 100
REPLAY_EVERY_N_GAMES = 20
BATCH_SIZE = REPLAY_EVERY_N_GAMES
N_ACTIONS = 8
EVAL_EVERY_N_EPISODES = 20
USE_ROLL_INSTEAD_OF_WIN_COUNT = False
PERSISTENT_STACKS = False
EVAL_AGAINST_RANDOM = False  # False = evaluates against older version (EVAL_EVERY_N_EPISODES episodes older)


def run_episode(agents):
    emulator = Emulator()
    temp_final_state = {}
    winner_counts = [0] * N_AGENTS
    n_games_played = 0
    for game in range(GAMES_PER_EPISODE):
        wrappers = []
        player_info = {}
        for i, agent in enumerate(agents):
            if PERSISTENT_STACKS:
                if temp_final_state:
                    for seat in temp_final_state:
                        player_info[seat.uuid] = {'name': 'Player ' + str(seat.uuid),
                                                  'stack': seat.stack if seat.stack else STACK_SIZE}
                else:
                    player_info[i] = {'name': 'Player ' + str(i), 'stack': STACK_SIZE}
            else:
                player_info[i] = {'name': 'Player ' + str(i), 'stack': STACK_SIZE}

            wrappers.append(DQNAgentWrapper(agent, STACK_SIZE))
            emulator.register_player(uuid=i, player=wrappers[-1])
        emulator.set_game_rule(N_AGENTS, 2, BB_SIZE / 2, 0)
        initial_game_state = emulator.generate_initial_game_state(player_info)

        game_state, events = emulator.start_new_round(initial_game_state)
        game_finish_state, events = emulator.run_until_round_finish(game_state)

        # import json
        # if game == 0 or game == 1:
        #     print('dumping')
        #     with open('event_dump_' + str(game), 'w') as f:
        #         json.dump(events, f, indent=2)
        if 'winners' not in events[-1]:
            events.pop()

        winner = events[-1]['winners'][0]['uuid']
        winner_counts[winner] += 1
        n_games_played += 1

        for i in range(N_AGENTS):
            reward = (game_finish_state['table'].seats.players[i].stack - wrappers[i].init_stack_size) / BB_SIZE
            # print('Starting stack:', wrappers[i].init_stack_size, 'Ending stack:', game_finish_state['table'].seats.players[i].stack, 'Reward:', reward)
            wrappers[i].agent.remember(wrappers[i].prev_state, wrappers[i].prev_action, reward, None, 1)


        temp_final_state = game_finish_state['table'].seats.players

        # print('====')
        print('\r{}'.format(game), end='')
        # print(game_finish_state)
        # print('\n')
        # print(events[-5:])
        # print('====')

        if (game % REPLAY_EVERY_N_GAMES == 0) or (game == GAMES_PER_EPISODE - 1):
            # replay memory for every agent
            for agent in agents:
                agent.replay(BATCH_SIZE)

        for i in range(N_AGENTS):
            agents[i].model.reset_states()

    for i in range(N_AGENTS):   # clear memory for fresh experience replay next episode
        agents[i].memory.clear()

    return agents, temp_final_state, winner_counts, n_games_played

def copy_agent(agent):
    weights = agent.model.get_weights()
    model = agent.model.get_config()
    del agent.model
    copied = copy.deepcopy(agent)
    agent.set_model(model, weights)
    copied.set_model(model, weights)
    return copied
    #return agent

if __name__ == '__main__':
    # used only for calculating # of features
    _sample_features = DQNAgent(3, 3, N_AGENTS).make_features(SAMPLE_ACTIONS, SAMPLE_HOLE_CARDS, SAMPLE_STATE)
    STATE_SIZE = len(_sample_features)

    oldest_agents = [DQNAgent(STATE_SIZE, N_ACTIONS, N_AGENTS)] * N_AGENTS
    old_agents = [DQNAgent(STATE_SIZE, N_ACTIONS, N_AGENTS)] * N_AGENTS
    agents = [DQNAgent(STATE_SIZE, N_ACTIONS, N_AGENTS)] * N_AGENTS

    if len(sys.argv) >= 3 and sys.argv[1] == '-l':  # load provided filename as weights
        for a in agents:
            a.load(sys.argv[2])

    hyperparam_list = {'games_per_episode': GAMES_PER_EPISODE, 'replay': REPLAY_EVERY_N_GAMES, 'n_episodes': N_EPISODES,
                       'n_agents': N_AGENTS, 'epsilon_min': agents[0].epsilon_min,
                       'epsilon_decay': agents[0].epsilon_decay,
                       'gamma': agents[0].gamma}
    print(hyperparam_list)

    for e in range(N_EPISODES):
        oldest_agents = [DQNAgent(STATE_SIZE, N_ACTIONS, N_AGENTS)] * N_AGENTS
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
        #new_agents[highest_idx].model.reset_states()# clear memory of RNN
        best_current_agent = copy_agent(new_agents[highest_idx])
        agents = [copy_agent(best_current_agent)] * N_AGENTS

        if EVAL_AGAINST_RANDOM:
            if e == N_EPISODES - 1 or e % EVAL_EVERY_N_EPISODES == 0:
                if e != 0:
                    print('====')
                    print('Final evaluation')
                    _, final_state, winner_counts, n_games_played = run_episode([best_current_agent] + oldest_agents[:-1])
                    print('\nNewest best won against oldest {} percent of games'.format(
                        (winner_counts[0] / n_games_played) * 100))
                    print('====')

        else:
            if e == N_EPISODES - 1 or e % EVAL_EVERY_N_EPISODES == 0:  # run 3x old versions against 1 new version
                if e != 0:
                    if e % 100 == 0:
                        print('====')
                        print('Final evaluation')
                        _, final_state, winner_counts, n_games_played = run_episode([best_current_agent] + oldest_agents[:-1])
                        print('\nNewest best won against oldest {} percent of games'.format(
                            (winner_counts[0] / n_games_played) * 100))
                        print('====')
                    else:
                        print('====')
                        print('Evaluating')
                        _, final_state, winner_counts, n_games_played = run_episode([best_current_agent] + old_agents[:-1])
                        print('\nNew won against old {} percent of games'.format(
                            (winner_counts[0] / n_games_played) * 100))
                        print('====')

                        for agent_idx in range(N_AGENTS):
                            old_agents[agent_idx] = copy_agent(best_current_agent)

    agents[0].save('pokerai_rl.h5')
