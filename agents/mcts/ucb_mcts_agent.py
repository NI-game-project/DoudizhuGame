import random
import timeit
from copy import deepcopy
from math import sqrt, log
import numpy as np
from envs.env import Env
from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1
from envs.doudizhu_rlcard import DoudizhuEnv
from utils_global import eval_tournament


class MCTSNode(object):

    def __init__(self, player_num, action, legal_actions, parent=None):
        #self.ucb_const = ucb_const
        self.action = action
        # list
        self.legal_actions = legal_actions
        self.parent = parent
        self.player_num = player_num
        # total rewards
        self.Q = [0.0 for i in range(player_num)]
        # visit sum
        self.N = [0.0 for i in range(player_num)]
        # a map from action to Node {key: action_id, value: MCTSTreeNode - child_node}
        self.children = {}

    def is_leaf(self):
        # check if this node is leaf(i.e. no nodes below this have been expanded).
        return self.children == {}

    def is_root(self):
        # check is this node is root(i.e. node has no parent).
        return self.parent is None

    def fully_expanded(self, player_id):
        # check whether all child nodes have been expanded.

        # for legal_action in self.legal_actions[player_id]:
        # if self.children.__contains__(legal_action):
        # return True
        if len(self.children) == len(self.legal_actions[player_id]):
            return True
        else:
            return False

    def add_child(self, child):
        self.children[child.action] = child

    def get_ucb_value(self, player_id, ucb_const):
        # Calculate and return the ucb1 value for this node.
        if self.N[player_id] == 0:
            return float('inf')
        else:
            # ucb1 formula = Q(v')/N(v') + C*(sqrt(2*ln(N(v)/N(v'))))
            # Q(v')/N(v') - value estimation, exploitation
            # C*(sqrt(2*ln(N(v)/N(v')))) - exploration
            ucb_value = self.Q[player_id] / self.N[player_id] + \
                        ucb_const * sqrt(2.0 * log(self.parent.N[player_id]) / self.N[player_id])
            return ucb_value

    def get_best_child(self, player_id, ucb_const):
        # best child = argmax(v'ϵ chilidren of v) (ucb_value)
        value_list = {action: child.get_ucb_value(player_id, ucb_const) for action, child in self.children.items()}
        action = max(value_list, key=value_list.get)
        child = self.children[action]
        return value_list, child

    def update(self, reward, player_id):
        # Update node values and visits.
        # reward: the payoff of subtree evaluation from the current player's perspective.
        self.Q[player_id] += reward
        self.N[player_id] += 1

    def __repr__(self):
        str = f'Node: {self.action}; children: {self.children}; visits: {self.N}; reward: {self.Q}'
        return str


class MCTSAgent:
    def __init__(self, env: Env, num_simulation):
        self.env = env
        self.use_raw = False
        self.num_simulations = num_simulation

    def step(self, state):

        legal_actions = state['legal_actions']
        current_player = self.env.get_player_id()
        all_legal_actions = [[] for i in range(self.env.player_num)]
        all_legal_actions[current_player] = legal_actions

        # initialize a root node from current state
        self.root = MCTSNode(player_num=self.env.player_num,
                             legal_actions=all_legal_actions, parent=None, action=0)

        temp_timestep = self.env.timestep
        for i in range(self.num_simulations):
            # Run a single random rollout from the root to the leaf,
            # get payoff at the leaf and propagate it back through its parents.
            # State is modified in-place, so a copy of environment must be provided.

            env_cp = deepcopy(self.env)
            selected_node = self.select(self.root, self.env)
            rewards = self.random_rollout(selected_node, self.env)
            self.back_prop(selected_node, rewards)

            for j in range(self.env.timestep - temp_timestep):
                self.env.step_back()
            self.env.timestep = temp_timestep
        # print(self.root)

        # get the value list and best child for the current root node(ucg_const=0), the child with best average score
        list, best_child = self.root.get_best_child(self.env.get_player_id(), 0)

        # return the action needed to reach the best child from the current root as action to take for the agent.
        action = best_child.action

        current_hand = state['raw_obs']['current_hand']
        print(f'current player: {current_player}, '
              f'current hand: {current_hand}, '
              f'action: {self.env._ACTION_LIST[action]}')

        return action

    def eval_step(self, state):
        probs = [0 for _ in range(self.env.action_num)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])
        return self.step(state), probs

    def select(self, root, env):
        player_id = env.get_player_id()
        # initialize a list of  for all the players
        unvisited_actions = [[] for i in range(env.player_num)]
        if not root.fully_expanded(player_id):
            unvisited_actions[player_id] = root.legal_actions[player_id]

        if len(unvisited_actions[player_id]) != 0:
            random.shuffle(unvisited_actions[player_id])
            untried_action = unvisited_actions[player_id].pop()
            node = self.expand(env, root, untried_action)

        else:
            _, node = root.get_best_child(player_id, 0.7)
            next_state, next_player_id = env.step(node.action)
            node.legal_actions[next_player_id] = next_state['legal_actions']

        return node

    def expand(self, env, node, action):
        # Instantiate one of the unexpanded children and return it.
        next_state, next_player_id = env.step(action)

        if node.children.__contains__(action):
            new_node = node.children[action]
            new_node.legal_actions[next_player_id] = next_state['legal_actions']
        else:
            legal_actions = [[] for i in range(env.player_num)]
            legal_actions[next_player_id] = next_state['legal_actions']
            new_node = MCTSNode(player_num=env.player_num, action=action, legal_actions=legal_actions, parent=node)

            node.add_child(new_node)

        return new_node

    def random_rollout(self, node, env):
        """
        random rollout policy: randomly selects action from legal_actions until game ends
        return:(list)  final reward
        """

        if env.is_over():
            # if game is over, return the playoff
            return env.get_payoffs()
        player_id = env.get_player_id()

        # randomly pick an action from current node's legal_actions
        action = np.random.choice(node.legal_actions[player_id])

        while not env.is_over():
            # step forward in the environment until game is over
            # execute the chosen action in env, get the next state and player id for next steps
            next_state, next_player_id = env.step(action)
            if not env.is_over():
                # if game is not over, next_action is randomly selected from legal_actions in next_state
                action = np.random.choice(next_state['legal_actions'])
        reward = env.get_payoffs()

        return reward

    def back_prop(self, node, rewards):

        while node is not None:

            for i in range(self.env.player_num):
                node.update(rewards[i], i)

            node = node.parent


def main():
    config = {
        'seed': None,
        'use_conv': False,
        'allow_step_back': True,
        'allow_raw_data': True,
        'record_action': True,
        'single_agent_mode': False,
        'active_player': None,
    }

    start = timeit.default_timer()

    # Make environment
    eval_env = DoudizhuEnv(config)
    evaluate_num = 1000

    # Set up the agents
    agent = MCTSAgent(env=eval_env, num_simulation=100)
    rule_agent = DouDizhuRuleAgentV1(action_num=eval_env.action_num)

    eval_env.set_agents([agent, rule_agent, rule_agent])

    payoff = eval_tournament(eval_env, evaluate_num)[0]
    print(payoff)
    print(f'time: {timeit.default_timer() - start} for {evaluate_num} episodes')


if __name__ == "__main__":
    main()
