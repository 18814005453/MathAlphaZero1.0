# core/engine.py
"""
AlphaZero MCTS 引擎 - 符号积分（最终稳定版）
修复：
- 反向传播折现错误（价值崩塌）
- 渐进扩宽误杀正确路径（死树问题）
- 双重温度指数放大
- 噪声注入后排序失步
- 对接 MathNet 的全局动作空间映射（修复训练对齐 Bug）
- 循环检测历史传递修复
- get_trajectory 完整实现
"""
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from core.state import IntegrationState   # 正确导入外部定义的状态类
from core.actions import Action, NUM_ACTIONS
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
            dirichlet_alpha: float = 0.3,
            dirichlet_epsilon: float = 0.25,
            progressive_widening_k: float = 5.0,
            progressive_widening_alpha: float = 0.5,
            max_depth: int = 30,
            device: str = 'cpu'  # 这里确保默认是 'cpu'
    ):
        self.network = network
        self.preprocessor = preprocessor
        self.env = IntegrationEnv()
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.pw_k = progressive_widening_k
        self.pw_alpha = progressive_widening_alpha
        self.max_depth = max_depth
        self.device = device
        self.root = None

    # ---------- 辅助方法：创建带有历史记录的新状态 ----------
    def _create_next_state(self, parent_state: IntegrationState, next_expr) -> IntegrationState:
        """根据父状态和下一步表达式生成新状态，并更新历史哈希"""
        new_history = parent_state.history_hashes.copy()
        new_history.add(parent_state.canonical_hash())
        return IntegrationState(
            expr=next_expr,
            depth=parent_state.depth + 1,
            history_hashes=new_history
        )

    # ------------------------------------------------------------
    def _expand_node(self, node: Node, add_dirichlet: bool = False):
        """获取所有合法动作及网络先验，全局排序，可选添加 Dirichlet 噪声"""
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

            # 1. 传入真实的表达式 .expr
            s# 1. 提取原始表达式张量
        raw_tensor = self.preprocessor.state_to_tensor(node.state.expr)

        # 2. 维度防御：严格规范化为 2D 张量 [batch_size, seq_len]
        if raw_tensor.dim() == 1:
            # 如果是 1D [seq_len]，加上 batch 维度
            state_tensor = raw_tensor.unsqueeze(0).to(self.device)
        elif raw_tensor.dim() == 3:
            # 如果不小心套多了变成 3D [1, 1, seq_len]，剥掉一层
            state_tensor = raw_tensor.squeeze(0).to(self.device)
        else:
            # 如果已经是 2D，直接转移到设备
            state_tensor = raw_tensor.to(self.device)

        # 3. 执行神经网络推理
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            # 输出如果是 [1, num_actions]，去掉 batch 维适配后续计算
            if policy_logits.dim() > 1:
                policy_logits = policy_logits.squeeze(0)

        node.value = value.item()

        legal_ids = [act.id for act in legal_acts]
        legal_logits = policy_logits[legal_ids]
        probs = F.softmax(legal_logits, dim=0).cpu().numpy().astype(np.float32)

        pairs = list(zip(legal_acts, probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        node.sorted_actions = [p[0] for p in pairs]
        node.sorted_probs = np.array([p[1] for p in pairs], dtype=np.float32)

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

    # ------------------------------------------------------------
    def _progressive_expand(self, node: Node) -> bool:
        """
        渐进扩宽：按先验排序顺序解锁子节点，直到达到 k 个有效子节点或动作耗尽。
        返回是否至少创建了一个新子节点。
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

            # 环境步进，获得原始 next_state（历史未更新）
            next_state_raw, reward, done, info = self.env.step(node.state, act)

            # 【关键修复】用正确的历史记录重新构建 next_state
            next_state = self._create_next_state(node.state, next_state_raw.expr)

            # 循环检测：如果新状态哈希已存在于父状态的历史中，跳过此动作
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
        """返回叶子节点及路径（不含叶子自身）"""
        path = []
        for _ in range(1000):
            if node.children:
                best_score = -float('inf')
                best_act = None
                best_child = None
                total_n = node.n
                for act, child in node.children.items():
                    u = self.c_puct * child.prior_p * math.sqrt(total_n + 1e-8) / (child.n + 1e-8)
                    score = child.q + u
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

            has_new = self._progressive_expand(node)
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
        """执行一次模拟，反向传播（修复折现错误）"""
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
            value_to_parent = self.gamma * accumulated + shaping
            parent.update(value_to_parent)
            accumulated = value_to_parent

    # ------------------------------------------------------------
    def search(self, state: IntegrationState) -> Dict[Action, int]:
        self.root = Node(state=state)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)

        for _ in range(self.num_simulations):
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
    def get_trajectory(self, state: IntegrationState, temperature: float = 1.0) -> List[dict]:
        """生成一条完整轨迹，用于训练数据收集"""
        trajectory = []
        cur_state = state

        for _ in range(self.max_depth):
            action_counts = self.search(cur_state)
            if not action_counts:
                break

            # 构建全局策略目标（密集向量）
            global_policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            total_n = sum(action_counts.values())
            for act, n in action_counts.items():
                prob = (n / total_n) ** (1.0 / temperature) if temperature > 0 else 0
                global_policy_target[act.id] = prob
            if global_policy_target.sum() > 0:
                global_policy_target /= global_policy_target.sum()
            else:
                # 兜底：选择访问次数最多的动作
                best_act = max(action_counts.items(), key=lambda x: x[1])[0]
                global_policy_target[best_act.id] = 1.0

            # 采样实际执行的动作
            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions])
            probs = counts / counts.sum()
            action = np.random.choice(actions, p=probs)

            # 记录当前步
            trajectory.append({
                "state": cur_state,
                "policy_target": global_policy_target.copy(),
                "value_target": self.root.q,   # 当前根节点的价值
                "action": action
            })

            # 执行动作，进入下一个状态（同时更新历史记录）
            next_state_raw, reward, done, info = self.env.step(cur_state, action)
            next_state = self._create_next_state(cur_state, next_state_raw.expr)

            # 如果游戏结束或达到最大深度则停止
            if done or next_state.depth >= self.max_depth:
                # 可选：添加最终价值
                break

            cur_state = next_state

        return trajectory