SAMPLE_STATE = {
  'round_count': 2,
  'dealer_btn': 1,
  'small_blind_pos': 2,
  'big_blind_pos': 0,
  'next_player': 0,
  'small_blind_amount': 10,
  'street': 'turn',
  'community_card': ['DJ', 'H6', 'S6', 'H5'],
  'seats': [
    {'stack': 95, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
    {'stack': 20, 'state': 'participating', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
    {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
  ],
  'pot': {
    'main': {'amount': 165},
    'side': [{'amount': 20, 'eligibles': ['ftwdqkystzsqwjrzvludgi', 'bbiuvgalrglojvmgggydyt']}]
  },
  'action_histories': {
    'preflop': [
      {'action': 'ANTE', 'amount': 5, 'uuid': 'zkbpehnazembrxrihzxnmt'},
      {'action': 'ANTE', 'amount': 5, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
      {'action': 'ANTE', 'amount': 5, 'uuid': 'bbiuvgalrglojvmgggydyt'},
      {'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'zkbpehnazembrxrihzxnmt'},
      {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
      {'action': 'CALL', 'amount': 20, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 20},
      {'action': 'RAISE', 'amount': 30, 'add_amount': 10, 'paid': 20, 'uuid': 'zkbpehnazembrxrihzxnmt'},
      {'action': 'CALL', 'amount': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 10},
      {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 10}
    ],
    'flop': [
      {'action': 'CALL', 'amount': 0, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 0},
      {'action': 'RAISE', 'amount': 30, 'add_amount': 30, 'paid': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
      {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 30},
      {'action': 'CALL', 'amount': 20, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 20}
     ],
    'turn': [],
  }
}

SAMPLE_HOLE_CARDS = ['DA', 'SA']
SAMPLE_ACTIONS = [
  {'action': 'fold', 'amount': 0},
  {'action': 'call', 'amount': 0},
  {'action': 'raise', 'amount': {'max': 95, 'min': 20}}
]