"""
AlphaZero MCTS 引擎 - 生产加固版（解决对齐、崩溃、性能、训练兼容性问题）
完整实现，可直接替换原有 core/engine.py
"""

import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from core.state import IntegrationState
from core.actions import Action
from core.env import IntegrationEnv
from core.network import MathNet
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
            strict_rule_mapping: bool = True
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

        # 规则索引获取函数（兼容严格映射与快速模式）
        self._get_rule_index = network.get_rule_index
        if not strict_rule_mapping and hasattr(network, '_fast_get_rule_index'):
            self._get_rule_index = network._fast_get_rule_index

        # 严格模式下确保规则缓存可用
        if strict_rule_mapping:
            assert hasattr(network, 'get_rule_embeddings'), "网络缺少 get_rule_embeddings"
            assert network.get_rule_embeddings() is not None, "规则缓存未刷新，请先调用 network.refresh_rule_cache"

    # -------------------- 核心 MCTS 方法 --------------------
    def _expand_node(self, node: Node, add_dirichlet: bool = False):
        """扩展节点：先推理网络，再获取合法动作并构建子节点（修复执行顺序与维度问题）"""
        if node.is_expanded:
            return

        # 1. 检查终局
        done, reward = self.env.is_terminal(node.state)
        if done or node.state.depth >= self.max_depth:
            node.terminal = done
            node.terminal_reward = reward if done else 0.0
            node.is_expanded = True
            return

        # 2. 构造输入张量（确保带有 batch 维度）
        raw_tensor = self.preprocessor.state_to_tensor(node.state.expr)
        if raw_tensor.dim() == 0:
            state_tensor = raw_tensor.unsqueeze(0).unsqueeze(0)
        else:
            state_tensor = raw_tensor.view(1, -1)
        state_tensor = state_tensor.to(self.device)

        # 3. 先执行网络前向传播
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            # 显式切片，避免 squeeze 导致标量
            if policy_logits.dim() > 1:
                policy_logits = policy_logits[0, :]   # [1, N] -> [N]

        node.value = value.item()

        # 4. 获取合法动作及索引
        legal_acts = self.env.legal_actions(node.state)
        if not legal_acts:
            node.is_expanded = True
            return

        node.legal_cache_indices = [self._get_rule_index(act.id) for act in legal_acts]

        # 5. 提取合法动作的 logits 并 softmax
        legal_logits = policy_logits[node.legal_cache_indices]   # 一维索引安全
        probs = F.softmax(legal_logits, dim=0).cpu().numpy().astype(np.float32)

        # 6. 按概率排序存储
        pairs = list(zip(legal_acts, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        node.sorted_actions = [p[0] for p in pairs]
        node.sorted_probs = np.array([p[1] for p in pairs], dtype=np.float32)

        # 7. 可选：Dirichlet 噪声（仅根节点）
        if add_dirichlet and len(node.sorted_actions) > 0:
            num_actions = len(node.sorted_actions)
            noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
            node.sorted_probs = (1 - self.dirichlet_epsilon) * node.sorted_probs + self.dirichlet_epsilon * noise
            node.sorted_probs = node.sorted_probs / node.sorted_probs.sum()
            combined = list(zip(node.sorted_actions, node.sorted_probs))
            combined.sort(key=lambda x: x[1], reverse=True)
            node.sorted_actions, node.sorted_probs = zip(*combined)
            node.sorted_actions = list(node.sorted_actions)
            node.sorted_probs = np.array(node.sorted_probs, dtype=np.float32)

        node.is_expanded = True
        node.num_unlocked = 0

    def _progressive_expand(self, node: Node) -> bool:
        """渐进式扩展：根据访问次数逐步解锁子节点"""
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

            # 使用 env.step 获得完整状态（包含 depth/history）
            next_state_raw, reward, done, info = self.env.step(node.state, act)
            next_state = next_state_raw

            # 环路检测（避免重复访问）
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

    def _select(self, node: Node) -> Tuple[Node, List[Tuple[Node, Action]]]:
        """选择阶段：使用 UCB 公式，并加入环路熔断"""
        path = []
        visited_hashes = set()
        for _ in range(1000):  # 防无限循环
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

                # 环路检测：若子节点状态已访问过则停止
                child_hash = best_child.state.canonical_hash()
                if child_hash in visited_hashes:
                    break
                visited_hashes.add(child_hash)

                path.append((node, best_act))
                node = best_child
                continue

            # 无子节点时尝试扩展
            if not node.is_expanded:
                self._expand_node(node)
            if node.terminal:
                break

            self._progressive_expand(node)
            if node.children:
                continue
            else:
                # 无法扩展则标记为终止
                if node.num_unlocked >= len(node.sorted_actions):
                    node.terminal = True
                    node.terminal_reward = -0.1
                break

        return node, path

    def _simulate(self):
        """一次模拟：选择→扩展→评估→回溯"""
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

    def search(self, state: IntegrationState) -> Dict[Action, int]:
        """执行全部模拟，返回子节点访问计数"""
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

    def get_action_probs(self, state: IntegrationState, temperature: float = 1.0) -> Tuple[List[Action], List[float]]:
        """获取动作概率分布（用于训练策略目标）"""
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
        """选择动作（推理/落子用）"""
        actions, probs = self.get_action_probs(state, temperature)
        if not actions:
            return None
        if temperature == 0:
            return actions[np.argmax(probs)]
        else:
            return np.random.choice(actions, p=probs)

    def get_trajectory(self, state: IntegrationState, temperature: float = 1.0,
                       include_metadata: bool = True) -> List[dict]:
        """生成一条完整轨迹（用于训练数据收集）"""
        trajectory = []
        cur_state = state
        total_start = time.time()

        for step_idx in range(self.max_depth):
            if self.timeout is not None and (time.time() - total_start) > self.timeout:
                print(f"⚠️ get_trajectory 整体超时 ({self.timeout:.1f}秒)，提前结束轨迹（已走 {step_idx} 步）")
                break

            action_counts = self.search(cur_state)
            if not action_counts:
                break

            # 动态温度：前8步用较高探索，之后降温
            current_temp = temperature if step_idx < 8 else 0.1
            current_rule_count = self.network.get_rule_embeddings().size(0)
            global_policy_target = np.zeros(current_rule_count, dtype=np.float32)

            total_n = sum(action_counts.values())
            for act, n in action_counts.items():
                prob = (n / total_n) ** (1.0 / current_temp) if current_temp > 0 else 0
                target_idx = self._get_rule_index(act.id)
                global_policy_target[target_idx] = prob

            if global_policy_target.sum() > 0:
                global_policy_target /= global_policy_target.sum()
            else:
                best_act = max(action_counts.items(), key=lambda x: x[1])[0]
                best_idx = self._get_rule_index(best_act.id)
                global_policy_target[best_idx] = 1.0

            # 按访问计数采样动作
            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
            probs = counts / counts.sum()
            if not np.isclose(probs.sum(), 1.0, atol=1e-8):
                probs = np.ones_like(probs) / len(probs)
            action = np.random.choice(actions, p=probs)

            entry = {
                "state": cur_state,
                "policy_target": global_policy_target.copy(),
                "value_target": self.root.q,
                "action": action
            }
            if include_metadata:
                entry["rule_count"] = current_rule_count
            trajectory.append(entry)

            # 执行动作，获取下一状态
            next_state_raw, reward, done, info = self.env.step(cur_state, action)
            next_state = next_state_raw

            if done or next_state.depth >= self.max_depth:
                break

            cur_state = next_state

        return trajectory