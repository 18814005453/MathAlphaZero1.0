"""
AlphaZero MCTS 引擎 - 生产加固版（解决对齐、崩溃、性能、训练兼容性问题）
"""

import math
import time
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from core.state import IntegrationState
from core.actions import Action
from core.env import IntegrationEnv
from core.network import MathNet
from core.rules import RULE_NAMES  # 仅作后备，推荐使用联合接口
from utils.preprocessor import MathPreprocessor


@dataclass
class Node:
    state: IntegrationState
    parent: Optional['Node'] = None
    action: Optional[Action] = None
    prior_p: float = 0.0
    n: int = 0
    w: float = 0.0
    q: float = 0.0
    children: Dict[Action, 'Node'] = field(default_factory=dict)
    is_expanded: bool = False
    sorted_actions: List[Action] = field(default_factory=list)
    sorted_probs: List[float] = field(default_factory=list)
    # 新增：缓存合法动作对应的网络输出行索引，避免重复调用 get_rule_index
    legal_cache_indices: List[int] = field(default_factory=list)
    num_unlocked: int = 0
    terminal: bool = False
    terminal_reward: float = 0.0
    value: float = 0.0
    heuristic_reward: float = 0.0

    def update(self, value: float):
        self.n += 1
        self.w += value
        self.q = self.w / self.n


class MCTS:
    def __init__(
            self,
            network: MathNet,
            preprocessor: MathPreprocessor,
            num_simulations: int = 50,
            c_puct: float = 1.0,
            gamma: float = 0.95,
            step_penalty: float = 0.05,
            dirichlet_alpha: float = 0.3,
            dirichlet_epsilon: float = 0.25,
            progressive_widening_k: float = 5.0,
            progressive_widening_alpha: float = 0.5,
            max_depth: int = 30,
            device: str = 'cpu',
            timeout: Optional[float] = None,
            strict_rule_mapping: bool = True  # 新增：是否严格检查规则映射（开发True，生产False）
    ):
        self.network = network
        self.preprocessor = preprocessor
        self.env = IntegrationEnv()
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
        self.strict_rule_mapping = strict_rule_mapping
        self.root = None

        # 预先获取规则映射函数（优化高频调用）
        self._get_rule_index = network.get_rule_index
        if not strict_rule_mapping and hasattr(network, '_fast_get_rule_index'):
            self._get_rule_index = network._fast_get_rule_index

        # 可选：验证缓存已刷新
        if strict_rule_mapping:
            assert hasattr(network, 'get_rule_embeddings'), "网络缺少 get_rule_embeddings"
            assert network.get_rule_embeddings() is not None, "规则缓存未刷新，请先调用 network.refresh_rule_cache"

    # ---------- 辅助方法：创建带有历史记录的新状态 ----------
    def _create_next_state(self, parent_state: IntegrationState, next_expr) -> IntegrationState:
        new_history = parent_state.history_hashes.copy()
        new_history.add(parent_state.canonical_hash())
        return IntegrationState(
            expr=next_expr,
            depth=parent_state.depth + 1,
            history_hashes=new_history
        )

    # ------------------------------------------------------------
    def _expand_node(self, node: Node, add_dirichlet: bool = False):
        """
        扩展节点：获取合法动作、网络先验，并一次性构建索引映射缓存。
        """
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

        # ---- 1. 预处理状态 ----
        raw_tensor = self.preprocessor.state_to_tensor(node.state.expr)
        if raw_tensor.dim() == 1:
            state_tensor = raw_tensor.unsqueeze(0).to(self.device)
        elif raw_tensor.dim() == 3:
            state_tensor = raw_tensor.squeeze(0).to(self.device)
        else:
            state_tensor = raw_tensor.to(self.device)

        # ---- 2. 网络前向（不传 current_num_actions）----
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            if policy_logits.dim() > 1:
                policy_logits = policy_logits.squeeze(0)

        node.value = value.item()

        # ---- 3. 【性能优化】一次性将合法动作转换为网络行索引并缓存到 node ----
        # 避免在后续渐进扩宽中重复调用 get_rule_index
        node.legal_cache_indices = [self._get_rule_index(act.id) for act in legal_acts]
        legal_logits = policy_logits[node.legal_cache_indices]
        probs = F.softmax(legal_logits, dim=0).cpu().numpy().astype(np.float32)

        # ---- 4. 排序 & Dirichlet 噪声 ----
        pairs = list(zip(legal_acts, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        node.sorted_actions = [p[0] for p in pairs]
        node.sorted_probs = np.array([p[1] for p in pairs], dtype=np.float32)

        if add_dirichlet and len(node.sorted_actions) > 0:
            num_actions = len(node.sorted_actions)
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
            node.sorted_probs = (1 - self.dirichlet_epsilon) * node.sorted_probs + self.dirichlet_epsilon * noise
            node.sorted_probs = node.sorted_probs / node.sorted_probs.sum()

            # 重新排序（噪声可能改变顺序）
            combined = list(zip(node.sorted_actions, node.sorted_probs))
            combined.sort(key=lambda x: x[1], reverse=True)
            node.sorted_actions, node.sorted_probs = zip(*combined)
            node.sorted_actions = list(node.sorted_actions)
            node.sorted_probs = np.array(node.sorted_probs, dtype=np.float32)

        node.is_expanded = True
        node.num_unlocked = 0

    # ------------------------------------------------------------
    def _progressive_expand(self, node: Node) -> bool:
        """
        渐进扩宽：使用 node 中已缓存的排序动作列表和概率。
        """
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

            next_state_raw, reward, done, info = self.env.step(node.state, act)
            next_state = self._create_next_state(node.state, next_state_raw.expr)

            # 循环检测
            if next_state.canonical_hash() in node.state.history_hashes:
                node.num_unlocked += 1
                continue

            child = Node(
                state=next_state,
                parent=node,
                action=act,
                prior_p=prob
            )
            if done:
                child.terminal = True
                child.terminal_reward = reward
            if 'heuristic_reward' in info:
                child.heuristic_reward = info['heuristic_reward']

            node.children[act] = child
            node.num_unlocked += 1
            unlocked += 1
            newly_created = True

        return newly_created

    # ------------------------------------------------------------
    def _select(self, node: Node) -> Tuple[Node, List[Tuple[Node, Action]]]:
        path = []
        for _ in range(1000):
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

    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    def search(self, state: IntegrationState) -> Dict[Action, int]:
        self.root = Node(state=state)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)

        start_time = time.time()
        for sim in range(self.num_simulations):
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                print(f"⚠️ MCTS 搜索超时 ({self.timeout:.1f}秒)，提前终止（已完成 {sim}/{self.num_simulations} 次模拟）")
                break
            self._simulate()

        return {act: child.n for act, child in self.root.children.items()}

    # ------------------------------------------------------------
    def get_action_probs(self, state: IntegrationState, temperature: float = 1.0) -> Tuple[List[Action], List[float]]:
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

    def choose_action(self, state: IntegrationState, temperature: float = 0.0) -> Optional[Action]:
        actions, probs = self.get_action_probs(state, temperature)
        if not actions:
            return None
        if temperature == 0:
            return actions[np.argmax(probs)]
        else:
            return np.random.choice(actions, p=probs)

    # ------------------------------------------------------------
    def get_trajectory(self, state: IntegrationState, temperature: float = 1.0,
                       include_metadata: bool = True) -> List[dict]:
        """
        生成完整轨迹，解决训练时维度不匹配问题。

        参数：
            include_metadata: 是否在轨迹中保存当前规则数（用于训练时零填充）

        返回的每个字典包含：
            - state, action, value_target
            - policy_target: 长度 = current_rule_count 的 numpy 数组
            - rule_count: (可选) 当时的规则总数
        """
        trajectory = []
        cur_state = state
        total_start = time.time()

        for step_idx in range(self.max_depth):
            if self.timeout is not None and (time.time() - total_start) > self.timeout:
                print(f"⚠️ get_trajectory 整体超时 ({self.timeout:.1f}秒)，提前结束轨迹（已走 {step_idx} 步）")
                break

            action_counts = self.search(cur_state)

            # 【修复2】防止空字典导致 max() 崩溃
            if not action_counts:
                break

            current_temp = temperature if step_idx < 8 else 0.1

            # 动态获取当前规则数 N
            current_rule_count = self.network.get_rule_embeddings().size(0)
            global_policy_target = np.zeros(current_rule_count, dtype=np.float32)

            total_n = sum(action_counts.values())
            for act, n in action_counts.items():
                prob = (n / total_n) ** (1.0 / current_temp) if current_temp > 0 else 0
                target_idx = self._get_rule_index(act.id)
                global_policy_target[target_idx] = prob

            # 归一化
            if global_policy_target.sum() > 0:
                global_policy_target /= global_policy_target.sum()
            else:
                # 【修复2加固】此时 action_counts 非空，取最大访问动作
                best_act = max(action_counts.items(), key=lambda x: x[1])[0]
                best_idx = self._get_rule_index(best_act.id)
                global_policy_target[best_idx] = 1.0

            # ---------- 安全采样 ----------
            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
            probs = counts / counts.sum()
            probs /= probs.sum()
            if not np.isclose(probs.sum(), 1.0, atol=1e-8):
                probs = np.ones_like(probs) / len(probs)
            action = np.random.choice(actions, p=probs)

            # 构建轨迹条目
            entry = {
                "state": cur_state,
                "policy_target": global_policy_target.copy(),
                "value_target": self.root.q,
                "action": action
            }
            if include_metadata:
                entry["rule_count"] = current_rule_count  # 用于训练时动态填充
            trajectory.append(entry)

            # 环境步进
            next_state_raw, reward, done, info = self.env.step(cur_state, action)
            next_state = self._create_next_state(cur_state, next_state_raw.expr)

            if done or next_state.depth >= self.max_depth:
                break

            cur_state = next_state

        return trajectory