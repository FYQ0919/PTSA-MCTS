import numpy as np
import torch
import math
import scipy.stats
import datetime


class MinMaxStats(object):

  def __init__(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound

  def update(self, value: float):
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    elif self.maximum == self.minimum:
      return 1.0
    return value

  def reset(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound


class Node(object):

  def __init__(self, prior):
    self.hidden_state = None
    self.visit_count = 0
    self.value_sum = 0
    self.reward = 0
    self.children = {}
    self.prior = prior
    self.to_play = 1
    self.aggregation_times = 0
    self.last_policy = None
    self.best_a = None
    self.block = False

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expand(self, network_output, to_play, actions, config):
    self.to_play = to_play
    self.hidden_state = network_output.hidden_state
    actions = np.array(actions)

    if network_output.reward:
      self.reward = network_output.reward.item()

    sample_num = config.num_sample_action

    policy_logits = network_output.policy_logits

    policy_values = torch.softmax(
      torch.tensor([policy_logits[0][a].item() for a in actions]), dim=0
    ).numpy().astype('float64')

    policy_values /= policy_values.sum()

    self.best_a = actions[np.argmax(policy_values)]

    if sample_num > 0:

      if len(actions) > sample_num:
        sample_action = np.random.choice(actions, size=sample_num, replace=False, p=policy_values)
      else:
        sample_action = actions

      sample_policy_values = torch.softmax(
        torch.tensor([policy_logits[0][a] for a in sample_action]), dim=0
      ).numpy().astype('float64')

      for i in range(len(sample_action)):
        a = sample_action[i]
        p = sample_policy_values[i]
        self.children[a] = Node(p)

    else:
      policy = {a: policy_values[i] for i, a in enumerate(actions)}

      for action, p in policy.items():
        self.children[action] = Node(p)
        self.children[action].last_policy = policy_values

  def add_exploration_noise(self, dirichlet_alpha, frac):
    actions = list(self.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
      self.children[a].prior = self.children[a].prior*(1-frac) + n*frac

class MCTS(object):

  def __init__(self, config):
    self.num_simulations = config.num_simulations
    self.config = config
    self.discount = config.discount
    self.pb_c_base = config.pb_c_base
    self.pb_c_init = config.pb_c_init
    self.init_value_score = config.init_value_score
    self.action_space = range(config.action_space)
    self.two_players = config.two_players
    self.known_bounds = config.known_bounds

    self.min_max_stats = MinMaxStats(*config.known_bounds)

  def run(self, root, network):
    self.min_max_stats.reset(*self.known_bounds)
    self.abstract_alpha = self.config.abstract_alpha

    search_paths = []

    for l in range(self.num_simulations):

      node = root
      search_path = [node]
      to_play = root.to_play

      while node.expanded():
        action, node = self.select_child(node)
        search_path.append(node)

        if self.two_players:
          to_play *= -1

      parent = search_path[-2]

      network_output = network.recurrent_inference(parent.hidden_state, [action])
      node.expand(network_output, to_play, self.action_space, self.config)

      self.backpropagate(search_path, network_output.value.item(), to_play)

      if search_path not in search_paths:

        search_paths.append(search_path)

      if self.abstract_alpha > 0 and len(search_paths) > 1:
        delet_paths = []

        for i in range(len(search_paths)-1):
          branch1_len = len(search_path)
          branch2 = search_paths[i]
          different_nodes = [[],[]]
          if branch1_len == len(branch2):

            branch_value_loss = 0
            aggregation_flag = True

            for j in range(1,branch1_len):
              if search_path[j] != branch2[j]:
                if search_path[j].visit_count > 0 and branch2[j].visit_count > 0:
                  is_aggregation, value_loss = self.abstract(search_path[j], branch2[j], type=self.config.abstract_type)
                  if not is_aggregation:
                    aggregation_flag = False
                    break
                  branch_value_loss += value_loss
                  different_nodes[0].append(search_path[j])
                  different_nodes[1].append(branch2[j])
                else:
                  aggregation_flag = False
                  break
              else:
                continue
            if aggregation_flag and len(different_nodes[0]) > 0:

               root.aggregation_times += 1

               if branch_value_loss >= 0:
                   delet_index, abstract_index = 1, 0
               else:
                  delet_index, abstract_index = 0, 1


               delet_node = different_nodes[delet_index][0]
               visit_count = delet_node.visit_count
               value_sum = delet_node.value_sum
               abstract_node = different_nodes[abstract_index][0]

               abstract_node.visit_count += visit_count
               abstract_node.value_sum += value_sum
               delet_node.block = True

        for path in delet_paths:
          search_paths.remove(path)

    return search_paths

  def select_child(self, node):
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    if node.visit_count == 0:
      _, action, child = max(
          (child.prior, action, child)
          for action, child in node.children.items())
      return action, child
    else:
      for action, child in node.children.items():
        if not child.block:
          ucb_score = self.ucb_score(node, child)
          if ucb_score > max_score:
            max_score = ucb_score
            max_index_lst.append(action)
          elif ucb_score >= max_score - epsilon:
            # NOTE: if the difference is less than  epsilon = 0.000001, we random choice action from  max_index_lst
            max_index_lst.append(action)
        else:
          continue

    if len(max_index_lst) > 0:
      best_action = np.random.choice(max_index_lst)
      child = node.children[best_action]
    else:
      _, best_action, child = max(
        (child.prior, action, child)
        for action, child in node.children.items())
    return best_action, child

  def ucb_score(self, parent, child):
    pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
      value = -child.value() if self.two_players else child.value()
      value_score = self.min_max_stats.normalize(child.reward + self.discount*value)
    else:
      value_score = self.init_value_score
    return prior_score + value_score

  def backpropagate(self, search_path, value, to_play):
    for idx, node in enumerate(reversed(search_path)):
      node.value_sum += value if node.to_play == to_play else -value
      node.visit_count += 1

      if self.two_players and node.to_play == to_play:
        reward = -node.reward
      else:
        reward = node.reward

      if idx < len(search_path) - 1:
        if self.two_players:
          new_q = node.reward - self.discount*node.value()
        else:
          new_q = node.reward + self.discount*node.value()
        self.min_max_stats.update(new_q)

      value = reward + self.discount*value

  def abstract(self, node1, node2, type):
    if type == 1:
      if node1.value() == node2.value() and node1.best_a == node2.best_a:
        value_loss = node1.value() - node2.value()
        return True, value_loss
      else:
        return False, 0

    elif type == 2:
      if abs(node1.value() - node2.value()) < self.abstract_alpha and node1.best_a == node2.best_a:
        value_loss = node1.value() - node2.value()
        return True, value_loss
      else:
        return False, 0

    elif type == 3:
      value_loss = node1.value() - node2.value()
      value1 = node1.value()
      value2 = node2.value()
      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:

        for a in node1.children.keys():
          if a in node2.children.keys():
           Q_sa = abs(value1 + self.discount * node1.children[a].reward - value2 + self.discount * node2.children[a].reward)
           if Q_sa > 0:
             return False, 0
        return True, value_loss
      else:
        return True, value_loss

    elif type == 4:

      value_loss = node1.value() - node2.value()
      value1 = node1.value()
      value2 = node2.value()

      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            Q_sa = abs(value1 + self.discount * node1.children[a].reward - value2 + self.discount * node2.children[a].reward)
            if Q_sa > self.abstract_alpha:
              return False, 0

        return True, value_loss
      else:
        return True, value_loss

    elif type == 5:
      value_loss = node1.value() - node2.value()
      value1 = node1.value()
      value2 = node2.value()
      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            
            if round(value1 + self.discount * node1.children[a].reward /self.abstract_alpha) != round(value2 + self.discount * node2.children[a].reward/self.abstract_alpha):
              return False, 0

        return True, value_loss
      else:
        return True, value_loss
    elif type == 6:
      value_loss = node1.value() - node2.value()
      q_dis1 = []
      q_dis2 = []
      value1 = node1.value()
      value2 = node2.value()

      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            q_dis1.append(value1 + self.discount * node1.children[a].reward)
            q_dis2.append(value2 + self.discount * node2.children[a].reward)
        if len(q_dis1) > 0 and len(q_dis2) > 0:
          q_dis1 = (np.array(q_dis1) - min(q_dis1) + 1e-3)
          q_dis2 = (np.array(q_dis2) - min(q_dis2) + 1e-3)

          q_dis1 = q_dis1 / np.sum(q_dis1)
          q_dis2 = q_dis2 / np.sum(q_dis2)

          p = self.abstract_alpha * (1 - np.clip(self.JS_loss(q_dis1, q_dis2), a_min=0, a_max=1))
          flag = np.random.choice([True, False], size=1, p=[p, 1 - p])
          return flag, value_loss if flag else 0
        else:
          return False, 0
      else:
        return True, value_loss

    elif type == 7:
      value_loss = node1.value() - node2.value()
      q_dis1 = []
      q_dis2 = []
      value1 = node1.value()
      value2 = node2.value()
      if len(node1.children.keys()) > 0 and len(node2.children.keys()) > 0:
        for a in node1.children.keys():
          if a in node2.children.keys():
            q_dis1.append(value1 + self.discount * node1.children[a].reward)
            q_dis2.append(value2 + self.discount * node2.children[a].reward)

        q_dis1 = np.exp(np.array(q_dis1))
        q_dis2 = np.exp(np.array(q_dis2))

        q_dis1 = q_dis1 / np.sum(q_dis1)
        q_dis2 = q_dis2 / np.sum(q_dis2)

        p = self.abstract_alpha * (1 - np.clip(self.JS_loss(q_dis1, q_dis2), a_min=0, a_max=1))
        flag = np.random.choice([True, False], size=1, p=[p, 1 - p])

        return flag, value_loss if flag else 0
      else:
        return True, value_loss

    else:
      return False, 0


  def JS_loss(self, dis1, dis2):
    average = (dis1 + dis2)/2

    return 0.5*scipy.stats.entropy(average,dis1,base=2) + 0.5*scipy.stats.entropy(average,dis2,base=2)



