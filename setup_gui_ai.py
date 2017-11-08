from agent.dqn_agent import DQNAgent
from environment.dqn_agent_wrapper import DQNAgentWrapper

N_AGENTS = 4
STATE_SIZE = 134
BB_SIZE = 10
STACK_SIZE = 200
N_ACTIONS = 8

def setup_ai(model_path):
    agent = DQNAgent(None, N_ACTIONS, N_AGENTS, None, None, None)
    agent.load(model_path)
    return DQNAgentWrapper(agent, STACK_SIZE)