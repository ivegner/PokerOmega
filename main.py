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
import argparse

import numpy as np
from keras.models import clone_model
from pypokerengine.api.emulator import Emulator

from agent.dqn_agent import DQNAgent
from environment.dqn_agent_wrapper import DQNAgentWrapper
from environment.sample_state import (SAMPLE_ACTIONS, SAMPLE_HOLE_CARDS,
                                      SAMPLE_STATE)

np.random.seed(12)

def parse_cli():
    parser = argparse.ArgumentParser(description='Train a PokerOmega instance with the given parameters')
    parser.add_argument('--agents', '-a', default='4', dest='n_agents',type=int, help = '(default: %(default)s)')
    parser.add_argument('--games', '-g', default='100', dest='games_per_episode', type=int, help = '(default: %(default)s)')
    parser.add_argument('--replay-every', '-r', dest='replay_every', default='20', type=int, help = '(default: %(default)s)')
    parser.add_argument('--episodes', '-e', dest='n_episodes', default='30', type=int, help = '(default: %(default)s)')
    parser.add_argument('--start-epsilon', '--starte', dest='start_e', default='1.0', type=float, help = '(default: %(default)s)')
    parser.add_argument('--epsilon-min', '--emin', dest='e_min', default='0.01', type=float, help = '(default: %(default)s)')
    parser.add_argument('--epsilon-decay', '--edec', dest='e_decay', default='0.995', type=float, help = '(default: %(default)s)')
    parser.add_argument('--gamma', default='0.95', dest='gamma', type=float, help = '(default: %(default)s)')
    parser.add_argument('--eval-every', '--eval', dest='eval_every', default='10', type=int, help = '(default: %(default)s)')
    parser.add_argument('--eval-against-random', '--random', dest='random_eval', default=False, action='store_true', help = '(default: %(default)s)')
    parser.add_argument('--load', '-l', dest='load', default=None,
                        help='Model weights file to restore. Will still initiate other vars to the other CLI params.')
    parser.add_argument('--output', '-o', dest='output_filename', default=None,
                        help='IF NOT SET, WILL NOT SAVE TRAINED MODEL. Automatically appends ".h5"')
    return parser.parse_args()

# Parse CLI args
args = parse_cli()
STATE_SIZE = 134
BB_SIZE = 10
STACK_SIZE = 200
N_ACTIONS = 8
USE_ROLL_INSTEAD_OF_WIN_COUNT = False
PERSISTENT_STACKS = False

N_AGENTS = args.n_agents
N_EPISODES = args.n_episodes
GAMES_PER_EPISODE = args.games_per_episode
REPLAY_EVERY_N_GAMES = args.replay_every
BATCH_SIZE = REPLAY_EVERY_N_GAMES
EVAL_EVERY_N_EPISODES = args.eval_every
EVAL_AGAINST_RANDOM = args.random_eval  # False = evaluates against older version (EVAL_EVERY_N_EPISODES episodes older)
STARTING_EPSILON = args.start_e
E_MIN = args.e_min
E_DECAY = args.e_decay
GAMMA = args.gamma

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

        temp_final_state = game_finish_state['table'].seats.players

        # print('====')
        print('\rGame:{}, epsilon:{}'.format(game, wrappers[0].agent.epsilon), end='')
        # print(game_finish_state)
        # print('\n')
        # print(events[-5:])
        # print('====')

        if (game % REPLAY_EVERY_N_GAMES == 0) or (game == GAMES_PER_EPISODE - 1):
            # replay memory for every agent
            # for agent in agents:
            #     agent.replay(BATCH_SIZE)
            agents[0].replay(BATCH_SIZE)

        for i in range(N_AGENTS):
            agents[i].model.reset_states()

    return agents[0], temp_final_state, winner_counts, n_games_played

def copy_agent(agent):
    weights = agent.model.get_weights()
    copied_model = clone_model(agent.model)
    copied = DQNAgent(*agent.get_init_info())
    copied.set_model(copied_model, weights)
    return copied
    #return agent

def make_random_agents():
    return [DQNAgent(STATE_SIZE, N_ACTIONS, N_AGENTS, STARTING_EPSILON, E_MIN, E_DECAY, GAMMA)] * N_AGENTS

# # used only for calculating # of features
# _sample_features = DQNAgent(3, 3, N_AGENTS).make_features(SAMPLE_ACTIONS, SAMPLE_HOLE_CARDS, SAMPLE_STATE)
# STATE_SIZE = len(_sample_features)

oldest_agents = make_random_agents()
old_agents = make_random_agents()
agents = make_random_agents()

# If load filename given, load weights
if args.load is not None:
    for a in agents:
        a.load(args.load)


hyperparam_list = {'games_per_episode': GAMES_PER_EPISODE, 'replay': REPLAY_EVERY_N_GAMES,
                   'n_episodes': N_EPISODES, 'n_agents': N_AGENTS, 'start_epsilon': agents[0].epsilon, 'epsilon_min': agents[0].epsilon_min,
                   'epsilon_decay': agents[0].epsilon_decay, 'gamma': agents[0].gamma}

print(hyperparam_list)

for e in range(N_EPISODES):
    oldest_agents = make_random_agents()
    new_agent, final_state, winner_counts, n_games_played = run_episode(agents)

    print('\nEpisode {} over'.format(e))
    best_current_agent = copy_agent(new_agent)
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

if args.output_filename:
    agents[0].save(args.output_filename + '.h5')
