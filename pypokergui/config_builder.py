import os

import yaml


def build_config(players=None, max_round=None, initial_stack=None, small_blind=None, ante=None, model_path=None):
    config = {
        "max_round": max_round,
        "initial_stack": initial_stack,
        "small_blind": small_blind,
        "ante": ante,
        "blind_structure": None,
        "ai_players": [],
        "model_path": model_path
        }
    path = os.path.abspath(model)
    for i in range(players):
        config['ai_players'].append({'name': 'Player_' + str(i), 'path': path})

    print(yaml.dump(config, default_flow_style=False))
