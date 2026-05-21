# core/engine.py
import math
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sympy import srepr, parse_expr

from core.state import IntegrationState
from core.actions import Action
from core.env import IntegrationEnv
from core.network import MathNet
from utils.preprocessor import MathPreprocessor
from knowledge.rule_registry import get_rule_id, get_all_rule_names, get_num_rules

class Node:
    def __init__(self, state: IntegrationState, parent=None, action=None, prior_p=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_p = prior_p
        self.n = 0
        self.w = 0.0
        self.q = 0.0
        self.children = {}
        self.is_expanded = False
        self.sorted_actions = []
        self.sorted_probs = []
        self.legal_cache_indices = []
        self.num_unlocked = 0
        self.terminal = False
        self.terminal_reward = 0.0
        self.value = 0.0
        self.heuristic_reward = 0.0

    def update(self, value):
        self.n += 1
        self.w += value
        self.q = self.w / self.n

class MCTS:
    def __init__(self, network: MathNet, preprocessor: MathPreprocessor,
                 num_simulations: int = 50, c_puct: float = 1.0,
                 gamma: float = 0.95, step_penalty: float = 0.05,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 progressive_widening_k: float = 5.0, progressive_widening_alpha: float = 0.5,
                 max_depth: int = 30, device: str = 'cpu', timeout: Optional[float] = None):
        self.network = network
        self.preprocessor = preprocessor
        self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout if timeout else 30.0)  # 复用 env 的超时
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.pw_k = progressive_widening_k
        self.pw_alpha = progressive_widening_alpha
        self.max_depth = max_depth
        self.device = device
        self.timeout = timeout
        self.root = None
        # 规则 ID 映射
        self.rule_name_to_id = {name: get_rule_id(name) for name in get_all_rule_names()}
        self.num_rules = get_num_rules()

    def _expand_node(self, node: Node, add_dirichlet: bool = False):
        if node.is_expanded:
            return
        done, reward = self.env.is_terminal(node.state)
        if done or node.state.depth >= self.max_depth:
            node.terminal = done
            node.terminal_reward = reward if done else 0.0
            node.is_expanded = True
            return

        legal_acts = self.env.legal_actions(node.state)
        if not legal_acts:
            node.is_expanded = True
            return

        # 构造掩码
        mask = torch.zeros(self.num_rules, dtype=torch.bool, device=self.device)
        for act in legal_acts:
            idx = self.rule_name_to_id.get(act.name, -1)
            if idx >= 0:
                mask[idx] = True

        # 标准化表达式
        try:
            norm_expr = parse_expr(srepr(node.state.expr))
        except:
            norm_expr = node.state.expr
        state_tensor = self.preprocessor.state_to_tensor(norm_expr).to(self.device)
        if state_tensor.dim() == 0:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        else:
            state_tensor = state_tensor.view(1, -1)

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor, mask)
            if policy_logits.dim() > 1:
                policy_logits = policy_logits[0, :]
        node.value = value.item()

        # 提取合法动作的概率
        legal_indices = [self.rule_name_to_id[act.name] for act in legal_acts]
        legal_logits = policy_logits[legal_indices]
        probs = F.softmax(legal_logits, dim=0).cpu().numpy().astype(np.float32)

        pairs = list(zip(legal_acts, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        node.sorted_actions = [p[0] for p in pairs]
        node.sorted_probs = np.array([p[1] for p in pairs], dtype=np.float32)
        node.legal_cache_indices = legal_indices

        if add_dirichlet and len(node.sorted_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.sorted_actions))
            node.sorted_probs = (1 - self.dirichlet_epsilon) * node.sorted_probs + self.dirichlet_epsilon * noise
            node.sorted_probs /= node.sorted_probs.sum()
            # 重新排序
            combined = list(zip(node.sorted_actions, node.sorted_probs))
            combined.sort(key=lambda x: x[1], reverse=True)
            node.sorted_actions, node.sorted_probs = zip(*combined)
            node.sorted_actions = list(node.sorted_actions)
            node.sorted_probs = np.array(node.sorted_probs)

        node.is_expanded = True
        node.num_unlocked = 0

    def _progressive_expand(self, node: Node) -> bool:
        if not node.is_expanded:
            self._expand_node(node)
        if node.terminal:
            return False
        total = len(node.sorted_actions)
        if total == 0:
            return False
        k = int(self.pw_k * ((node.n + 1) ** self.pw_alpha))
        k = min(max(k, 1), total)
        unlocked = len(node.children)
        if unlocked >= k:
            return False
        newly_created = False
        while unlocked < k and node.num_unlocked < total:
            idx = node.num_unlocked
            act = node.sorted_actions[idx]
            prob = node.sorted_probs[idx]
            next_state, reward, done, info = self.env.step(node.state, act)
            # 环路检测
            if next_state.canonical_hash() in node.state.history_hashes:
                node.num_unlocked += 1
                continue
            child = Node(next_state, parent=node, action=act, prior_p=prob)
            if done:
                child.terminal = True
                child.terminal_reward = reward
            if 'simplify_reward' in info:
                child.heuristic_reward = info['simplify_reward']
            node.children[act] = child
            node.num_unlocked += 1
            unlocked += 1
            newly_created = True
        return newly_created

    def _select(self, node: Node):
        path = []
        visited_hashes = set()
        while True:
            if node.children:
                best_score = -float('inf')
                best_act = None
                best_child = None
                total_n = node.n
                for act, child in node.children.items():
                    u = self.c_puct * child.prior_p * math.sqrt(total_n + 1e-8) / (child.n + 1e-8)
                    child_q = child.q if child.n > 0 else node.q
                    score = child_q + u
                    if score > best_score:
                        best_score = score
                        best_act = act
                        best_child = child
                child_hash = best_child.state.canonical_hash()
                if child_hash in visited_hashes:
                    break
                visited_hashes.add(child_hash)
                path.append((node, best_act))
                node = best_child
                continue
            if not node.is_expanded:
                self._expand_node(node)
            if node.terminal:
                break
            self._progressive_expand(node)
            if node.children:
                continue
            else:
                if node.num_unlocked >= len(node.sorted_actions):
                    node.terminal = True
                    node.terminal_reward = -0.1
                break
        return node, path

    def _simulate(self):
        leaf, path = self._select(self.root)
        if leaf.terminal:
            leaf_value = leaf.terminal_reward
        else:
            if not leaf.is_expanded:
                self._expand_node(leaf)
            leaf_value = leaf.value if hasattr(leaf, 'value') else 0.0
        leaf.update(leaf_value)
        accumulated = leaf_value
        for parent, act in reversed(path):
            child = parent.children[act]
            shaping = child.heuristic_reward
            value_to_parent = self.gamma * accumulated + shaping - self.step_penalty
            parent.update(value_to_parent)
            accumulated = value_to_parent

    def search(self, state: IntegrationState):
        self.root = Node(state)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)
        start_time = time.time()
        for sim in range(self.num_simulations):
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                break
            self._simulate()
        return {act: child.n for act, child in self.root.children.items()}

    def get_action_probs(self, state: IntegrationState, temperature: float = 1.0):
        action_counts = self.search(state)
        if not action_counts:
            return [], []
        actions = list(action_counts.keys())
        counts = np.array([action_counts[a] for a in actions])
        if temperature == 0:
            probs = (counts == counts.max()).astype(float)
        else:
            probs = counts ** (1.0 / temperature)
        probs /= probs.sum()
        return actions, probs.tolist()

    def choose_action(self, state: IntegrationState, temperature: float = 0.0):
        actions, probs = self.get_action_probs(state, temperature)
        if not actions:
            return None
        if temperature == 0:
            return actions[np.argmax(probs)]
        else:
            return np.random.choice(actions, p=probs)

    def get_trajectory(self, state: IntegrationState, temperature: float = 1.0,
                       include_metadata: bool = True) -> List[dict]:
        trajectory = []
        cur_state = state
        start_time = time.time()
        for step_idx in range(self.max_depth):
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                break
            action_counts = self.search(cur_state)
            if not action_counts:
                break
            current_temp = temperature if step_idx < 8 else 0.1
            global_policy_target = np.zeros(self.num_rules, dtype=np.float32)
            total_n = sum(action_counts.values())
            for act, n in action_counts.items():
                prob = (n / total_n) ** (1.0 / current_temp) if current_temp > 0 else 0
                idx = self.rule_name_to_id.get(act.name, -1)
                if idx >= 0:
                    global_policy_target[idx] = prob
            if global_policy_target.sum() > 0:
                global_policy_target /= global_policy_target.sum()
            else:
                best_act = max(action_counts.items(), key=lambda x: x[1])[0]
                best_idx = self.rule_name_to_id.get(best_act.name, -1)
                if best_idx >= 0:
                    global_policy_target[best_idx] = 1.0
                else:
                    global_policy_target = np.ones(self.num_rules) / self.num_rules
            # 采样动作
            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
            probs = counts / counts.sum()
            action = np.random.choice(actions, p=probs)
            # 记录轨迹
            entry = {
                "state": cur_state,
                "policy_target": global_policy_target.copy(),
                "value_target": self.root.q,
                "action": action,
                "q_values": {act.name: child.q for act, child in self.root.children.items()} if self.root.children else {}
            }
            if include_metadata:
                entry["rule_count"] = self.num_rules
            trajectory.append(entry)
            next_state, reward, done, info = self.env.step(cur_state, action)
            if done or next_state.depth >= self.max_depth:
                break
            cur_state = next_state
        return trajectory