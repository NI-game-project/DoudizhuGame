import os
import timeit
from datetime import datetime
import torch

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

### random or rule_based agents for evaluating ###
from agents.non_rl.dummyrule_agent import DummyRuleAgent as RandomAgent
from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
# from agents.non_rl.rhcp_agent import RHCPAgent as RuleAgent

### rl_agents for training ###
### uncomment these lines to import different Agent for training
#from agents.value_based.n_step_dqn import NStepDQNAgent as RLAgent
#from agents.value_based.per_agent import PERDQNAgent as RLAgent
#from agents.value_based.rainbow_c51 import RainbowAgent as RLAgent
#from agents.value_based.rainbow_qr import RainbowAgent as RLAgent
from agents.value_based.rainbow_mog import RainbowAgent as RLAgent

which_agent = 'rainbow_mog'
eval_every = 500
eval_num = 1000
episode_num = 100_000

save_dir = f'./experiments/{which_agent}/'
logger = Logger(save_dir)

save_every = 2000

# Make environments to train and evaluate models
config = {
    'seed': None,
    # add key 'use_conv' to config dict, if using mydoudizhu.py as env
    # to indicate whether using state_encoding for fc_net or conv_net.
    'use_conv': False,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}
state_shape = [9, 4, 15]
env_type = 'cooperation'
env = Env(config, state_shape=state_shape, type=env_type)
eval_env = Env(config, state_shape=state_shape, type=env_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize agent for evaluation

rule_agent = RuleAgent(action_num=eval_env.action_num)
rand_agent = RandomAgent(action_num=env.action_num)

# initialize rl_agents
agent_p1 = RLAgent(num_actions=env.action_num,
                   state_shape=env.state_shape,
                   )
agent_p2 = RLAgent(num_actions=env.action_num,
                   state_shape=env.state_shape,
                   )
env.set_agents([rule_agent, agent_p1, agent_p1])
eval_env.set_agents([rule_agent, agent_p1, agent_p2])
print(agent_p1.online_net)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num + 1):
    # get transitions by playing an episode in envs
    trajectories, _ = env.run(is_training=True)
    # train the agent against rule_based agent
    # set the agent's online network to training mode
    agent_p1.train_mode()
    agent_p2.train_mode()

    for trajectory in trajectories[1]:
        agent_p1.add_transition(trajectory)
    for trajectory in trajectories[2]:
        agent_p2.add_transition(trajectory)

    # evaluate against random agent
    if episode % eval_every == 0:
        print(datetime.now().strftime("%H:%M:%S"))
        # Set agent's online network to evaluation mode
        agent_p1.eval_mode()
        agent_p2.eval_mode()
        result, states = tournament(eval_env, eval_num, agent_p1)
        logger.log_performance(episode, result[1], agent_p1.loss, states[1][-1][0]['raw_obs'])

        print(f'\nepisode: {episode}, result: {result}')

    if episode % save_every == 0:
        agent_p1.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_p1_ckpt.pt'))
        agent_p2.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_p2_ckpt.pt'))
        now = datetime.now().strftime("%H:%M:%S")
        print(f'\nckpt save for ep:{episode} at {now}')

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start) / 3600} hrs for {episode_num} episodes')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('dqn.vs.rule')

# Save model
agent_p1.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_p1.pt'))
agent_p2.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_p1.pt'))
