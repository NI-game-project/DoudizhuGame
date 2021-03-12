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
#from agents.value_based.per_dqn_agent import PERDQNAgent as RLAgent
#from agents.value_based.nstep_noisy_duel_double_dqn import DQNAgent as RLAgent
#from agents.value_based.C51_dqn_agent import C51DQNAgent as RLAgent
#from agents.value_based.per_noisy_duel_double_dqn_agent import PERDQNAgent as RLAgent
from agents.value_based.rainbow_agent import RainbowAgent as RLAgent


which_run = '1'
which_agent = 'rainbow'
eval_every = 500
eval_num = 1000
episode_num = 100_000

save_dir = f'./experiments/{which_run}/{which_agent}/'
logger = Logger(save_dir)

best_result = 0

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
state_shape = [5, 4, 15]
env = Env(config, state_shape=state_shape)
eval_env = Env(config, state_shape=state_shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize agent for evaluation

rule_agent = RuleAgent(action_num=eval_env.action_num)
rand_agent = RandomAgent(action_num=env.action_num)

# initialize rl_agents
agent = RLAgent(num_actions=env.action_num,
                state_shape=env.state_shape,
                lr=.00001,
                gamma=0.97,
                batch_size=64,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=80000,
                soft_update=False,
                train_every=1,
                hard_update_target_every=1000,
                replay_memory_init_size=1000,
                replay_memory_size=int(2e5),
                clip=True,
                )
env.set_agents([agent, rule_agent, rule_agent])
eval_env.set_agents([agent, rule_agent, rule_agent])
print(agent.online_net)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num + 1):
    # get transitions by playing an episode in envs
    trajectories, _ = env.run(is_training=True)
    # train the agent against rule_based agent
    # set the agent's online network to training mode
    agent.train_mode()

    for trajectory in trajectories[0]:
        agent.add_transition(trajectory)

    # evaluate against random agent
    if episode % eval_every == 0:

        # Set agent's online network to evaluation mode
        agent.eval_mode()
        result, states = tournament(eval_env, eval_num, agent)
        logger.log_performance(episode, result[0], agent.loss, states[0][-1][0]['raw_obs'],
                               agent.actions, agent.predictions, agent.q_values,
                               agent.current_q_values, agent.expected_q_values)

        print(f'\nepisode: {episode}, result: {result}, '
              f'epsilon: {agent.epsilon}, '
              #f'lr: {agent.lr_scheduler.get_lr()}'
              )

        if result[0] > best_result:
            best_result = result[0]
            agent.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_landlord_best.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start) / 3600} hrs for {episode_num} episodes')
print(f'best_result: {best_result}')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('dqn.vs.rule')

# Save model
agent.save_state_dict(os.path.join(save_dir, f'{which_agent}_agent_landlord.pt'))
