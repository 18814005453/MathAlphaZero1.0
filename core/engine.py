# core/engine.py -- MathAlphaZero v6.0 (Deep MCTS)
import math
import time
import threading
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
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
    __slots__ = ('state', 'parent', 'action', 'prior_p', 'n', 'w', 'q', 'children',
                 'is_expanded', 'sorted_actions', 'sorted_probs',
                 'num_unlocked', 'terminal', 'terminal_reward', 'value', 'heuristic_reward',
                 'virtual_loss', 'lock', 'depth')
    def __init__(self, state: IntegrationState, parent=None, action=None, prior_p=0.0, depth=0):
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
        self.sorted_probs = np.array([], dtype=np.float32)
        self.num_unlocked = 0
        self.terminal = False
        self.terminal_reward = 0.0
        self.value = 0.0
        self.heuristic_reward = 0.0
        self.virtual_loss = 0
        self.depth = depth
        self.lock = threading.Lock()

    def update(self, value: float):
        with self.lock:
            self.n += 1
            self.w += value
            self.q = self.w / self.n

    def add_virtual_loss(self):
        with self.lock:
            self.virtual_loss += 1
            self.n += 1
            self.w -= 1.0
            self.q = self.w / self.n

    def revert_virtual_loss(self):
        with self.lock:
            self.virtual_loss -= 1
            self.n -= 1
            self.w += 1.0
            self.q = self.w / self.n if self.n > 0 else 0.0


class MCTS:
    """Deep MCTS: multi-step tree search for symbolic integration."""

    def __init__(self, network: MathNet, preprocessor: MathPreprocessor,
                 num_simulations=80, c_puct=1.0, gamma=0.95, step_penalty=0.02,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
                 pw_k=5.0, pw_alpha=0.5, max_depth=30, device='cpu',
                 timeout=None, num_parallel=1,
                 max_top_rules=10, max_top_positions=8,
                 tree_depth_limit=4):
        self.network = network
        self.preprocessor = preprocessor
        self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout or 30.0)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.pw_k = pw_k
        self.pw_alpha = pw_alpha
        self.max_depth = max_depth
        self.device = device
        self.timeout = timeout
        self.num_parallel = max(1, num_parallel)
        self.max_top_rules = max_top_rules
        self.max_top_positions = max_top_positions
        self.tree_depth_limit = tree_depth_limit
        self.root = None
        self.rule_name_to_id = {name: get_rule_id(name) for name in get_all_rule_names()}
        self.num_rules = get_num_rules()

    def _expand_node(self, node: Node, add_dirichlet=False):
        if node.is_expanded:
            return
        done, reward = self.env.is_terminal(node.state)
        if done or node.state.depth >= self.max_depth:
            node.terminal = done
            node.terminal_reward = reward if done else 0.0
            node.is_expanded = True
            return

        legal_actions = self.env.legal_actions(node.state)
        if not legal_actions:
            node.terminal = True
            node.terminal_reward = -0.05
            node.is_expanded = True
            return

        try:
            norm_expr = parse_expr(srepr(node.state.expr))
        except Exception:
            norm_expr = node.state.expr

        state_tensor, depth_tensor = self.preprocessor.state_to_tensor_with_depth(norm_expr)
        state_tensor = state_tensor.to(self.device)
        depth_tensor = depth_tensor.to(self.device) if depth_tensor is not None else None
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            if depth_tensor is not None:
                depth_tensor = depth_tensor.unsqueeze(0)

        rule_mask = torch.zeros(self.num_rules, dtype=torch.bool, device=self.device)
        for act in legal_actions:
            idx = self.rule_name_to_id.get(act.name, -1)
            if idx >= 0:
                rule_mask[idx] = True
        rule_mask = rule_mask.unsqueeze(0)

        seq_len = state_tensor.size(1)
        location_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        for act in legal_actions:
            if 0 <= act.pos < seq_len:
                location_mask[act.pos] = True
        location_mask = location_mask.unsqueeze(0)

        with torch.no_grad():
            rule_probs, location_probs, value = self.network.predict_rule_and_location(
                state_tensor, depth_tensor, rule_mask, location_mask)
        node.value = value.item()

        rule_probs_np = rule_probs[0].cpu().numpy()
        location_probs_np = location_probs[0].cpu().numpy()

        top_rule_indices = np.argsort(rule_probs_np)[-self.max_top_rules:][::-1]
        candidate_actions = []
        candidate_probs = []

        for r_idx in top_rule_indices:
            rule_id = self.network.idx_to_id.get(r_idx, None)
            if rule_id is None:
                continue
            rule_name = None
            for name, rid in self.rule_name_to_id.items():
                if rid == rule_id:
                    rule_name = name
                    break
            if rule_name is None:
                continue
            rule_prob = rule_probs_np[r_idx]
            pos_probs = location_probs_np.copy()
            valid_positions = [act.pos for act in legal_actions
                               if act.name == rule_name and 0 <= act.pos < seq_len]
            for p in range(seq_len):
                if p not in valid_positions:
                    pos_probs[p] = 0.0
            if np.sum(pos_probs) == 0:
                continue
            pos_probs = pos_probs / np.sum(pos_probs)
            top_pos_indices = np.argsort(pos_probs)[-self.max_top_positions:][::-1]
            for p in top_pos_indices:
                if pos_probs[p] <= 0:
                    continue
                joint_prob = rule_prob * pos_probs[p]
                act = Action(id=rule_id, name=rule_name, pos=int(p))
                candidate_actions.append(act)
                candidate_probs.append(joint_prob)

        if not candidate_actions:
            node.terminal = True
            node.terminal_reward = -0.05
            node.is_expanded = True
            return

        sorted_pairs = sorted(zip(candidate_actions, candidate_probs), key=lambda x: x[1], reverse=True)
        node.sorted_actions = [p[0] for p in sorted_pairs]
        node.sorted_probs = np.array([p[1] for p in sorted_pairs], dtype=np.float32)

        if add_dirichlet and len(node.sorted_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(node.sorted_actions))
            node.sorted_probs = (1 - self.dirichlet_epsilon) * node.sorted_probs + self.dirichlet_epsilon * noise
            node.sorted_probs /= node.sorted_probs.sum()
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
            if next_state.canonical_hash() in node.state.history_hashes:
                node.num_unlocked += 1
                continue
            child = Node(next_state, parent=node, action=act, prior_p=prob, depth=node.depth + 1)
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

    def _select(self, node: Node) -> Tuple[Node, List[Tuple[Node, Action]]]:
        """Select + auto-expand deep leaves. Returns (leaf_node, path)."""
        path = []
        visited = set()
        while True:
            if node.children:
                best_score = -float('inf')
                best_child = None
                best_act = None
                total_n = node.n
                for act, child in node.children.items():
                    u = self.c_puct * child.prior_p * math.sqrt(total_n + 1e-8) / (child.n + 1e-8)
                    q = child.q if child.n > 0 else node.q
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_act = act
                        best_child = child
                ch = best_child.state.canonical_hash()
                if ch in visited:
                    break
                visited.add(ch)
                path.append((node, best_act))
                node = best_child
                continue

            if not node.is_expanded:
                self._expand_node(node)

            if node.terminal:
                break

            # Deep expansion: if depth limit not reached, try to grow children
            if node.depth < self.tree_depth_limit:
                self._progressive_expand(node)
                if node.children:
                    continue

            # Leaf: no children and can't expand further
            if node.num_unlocked >= len(node.sorted_actions) and not node.children:
                node.terminal = True
                node.terminal_reward = -0.02
            break

        return node, path

    def _simulate(self) -> None:
        """Deep MCTS simulation through the tree."""
        leaf, path = self._select(self.root)

        if leaf.terminal:
            leaf_value = leaf.terminal_reward
        else:
            if not leaf.is_expanded:
                self._expand_node(leaf)
            leaf_value = leaf.value

        leaf.update(leaf_value)
        accumulated = leaf_value
        for parent, act in reversed(path):
            child = parent.children[act]
            shaping = child.heuristic_reward
            value_to_parent = self.gamma * accumulated + shaping - self.step_penalty
            parent.update(value_to_parent)
            accumulated = value_to_parent

    def search(self, state: IntegrationState) -> Dict[Action, int]:
        """Run MCTS search. Returns {action: visit_count} from root children."""
        self.root = Node(state, depth=0)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)

        start_time = time.time()
        for sim in range(self.num_simulations):
            if self.timeout and (time.time() - start_time) > self.timeout:
                break
            self._simulate()

        return {act: child.n for act, child in self.root.children.items()}

    def get_action_probs(self, state: IntegrationState, temperature=1.0):
        counts = self.search(state)
        if not counts:
            return [], []
        actions = list(counts.keys())
        arr = np.array([counts[a] for a in actions], dtype=np.float64)
        if temperature == 0:
            probs = (arr == arr.max()).astype(float)
        else:
            probs = arr ** (1.0 / temperature)
        probs /= probs.sum()
        return actions, probs.tolist()

    def choose_action(self, state: IntegrationState, temperature=0.0):
        actions, probs = self.get_action_probs(state, temperature)
        if not actions:
            return None
        if temperature == 0:
            return actions[np.argmax(probs)]
        return np.random.choice(actions, p=probs)

    def get_trajectory(self, state: IntegrationState, temperature=1.0, include_metadata=True):
        """Deep trajectory: MCTS searches multiple steps, building a tree at each step."""
        trajectory = []
        cur_state = state
        start_time = time.time()
        for step_idx in range(self.max_depth):
            if self.timeout and (time.time() - start_time) > self.timeout:
                break
            action_counts = self.search(cur_state)
            if not action_counts:
                break

            current_temp = temperature if step_idx < 8 else 0.1
            max_len = self.network.max_len
            rule_target = np.zeros(self.num_rules, dtype=np.float32)
            location_target = np.zeros(max_len, dtype=np.float32)
            total_n = sum(action_counts.values())
            for act, n in action_counts.items():
                prob = (n / total_n) ** (1.0 / current_temp) if current_temp > 0 else 0
                if prob > 0:
                    rule_idx = self.rule_name_to_id.get(act.name, -1)
                    if rule_idx >= 0:
                        rule_target[rule_idx] += prob
                    if 0 <= act.pos < max_len:
                        location_target[act.pos] += prob
            if rule_target.sum() > 0:
                rule_target /= rule_target.sum()
            if location_target.sum() > 0:
                location_target /= location_target.sum()
            else:
                location_target.fill(1.0 / max_len)

            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
            action = np.random.choice(actions, p=counts / counts.sum())

            entry = {
                "state": cur_state,
                "rule_policy_target": rule_target.copy(),
                "location_policy_target": location_target.copy(),
                "value_target": self.root.q if self.root else 0.0,
                "action": action,
                "q_values": {f"{act.name}@{act.pos}": child.q
                             for act, child in self.root.children.items()}
                if self.root and self.root.children else {}
            }
            if include_metadata:
                entry["rule_count"] = self.num_rules
                entry["max_len"] = self.network.max_len
            trajectory.append(entry)

            next_state, reward, done, info = self.env.step(cur_state, action)
            if done or next_state.depth >= self.max_depth:
                break
            cur_state = next_state

        return trajectory
