# core/engine.py
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
    """MCTS 节点，支持虚拟损失和线程安全"""
    __slots__ = ('state', 'parent', 'action', 'prior_p', 'n', 'w', 'q', 'children',
                 'is_expanded', 'sorted_actions', 'sorted_probs', 'legal_cache_indices',
                 'num_unlocked', 'terminal', 'terminal_reward', 'value', 'heuristic_reward',
                 'virtual_loss', 'lock')
    def __init__(self, state: IntegrationState, parent=None, action=None, prior_p=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_p = prior_p
        self.n = 0           # 访问次数
        self.w = 0.0         # 总价值
        self.q = 0.0         # 平均价值
        self.children = {}
        self.is_expanded = False
        self.sorted_actions = []      # 合法动作按先验概率降序
        self.sorted_probs = np.array([], dtype=np.float32)
        self.legal_cache_indices = [] # 对应的规则 ID 列表
        self.num_unlocked = 0         # 已扩展的子节点数量
        self.terminal = False
        self.terminal_reward = 0.0
        self.value = 0.0              # 网络估计的状态价值
        self.heuristic_reward = 0.0   # 启发式奖励（如化简奖励）
        self.virtual_loss = 0         # 虚拟损失计数
        self.lock = threading.Lock()  # 线程锁

    def update(self, value: float):
        """更新节点统计（线程安全）"""
        with self.lock:
            self.n += 1
            self.w += value
            self.q = self.w / self.n

    def add_virtual_loss(self):
        """添加虚拟损失（临时降低节点价值）"""
        with self.lock:
            self.virtual_loss += 1
            self.n += 1
            self.w -= 1.0   # 临时减去1，使 Q 值降低
            self.q = self.w / self.n

    def revert_virtual_loss(self, value: float):
        """恢复虚拟损失，并累加真实价值"""
        with self.lock:
            self.virtual_loss -= 1
            self.n -= 1
            self.w += 1.0 + value   # 移除临时减去的1，加上真实价值
            self.q = self.w / self.n


class MCTS:
    """
    蒙特卡洛树搜索，支持并行模拟、渐进扩展、虚拟损失和探索常数调度。
    """
    def __init__(
        self,
        network: MathNet,
        preprocessor: MathPreprocessor,
        num_simulations: int = 50,
        c_puct: float = 1.0,
        c_puct_decay: float = 0.0,        # 每次模拟后 c_puct 衰减因子（1 - decay）
        gamma: float = 0.95,              # 折扣因子
        step_penalty: float = 0.05,       # 步惩罚（已包含在环境奖励中，这里用于备份转移）
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        progressive_widening_k: float = 5.0,
        progressive_widening_alpha: float = 0.5,
        max_depth: int = 30,
        device: str = 'cpu',
        timeout: Optional[float] = None,
        num_parallel: int = 1             # 并行模拟线程数
    ):
        self.network = network
        self.preprocessor = preprocessor
        self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout if timeout else 30.0)
        self.num_simulations = num_simulations
        self.base_c_puct = c_puct
        self.c_puct = c_puct
        self.c_puct_decay = c_puct_decay
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.pw_k = progressive_widening_k
        self.pw_alpha = progressive_widening_alpha
        self.max_depth = max_depth
        self.device = device
        self.timeout = timeout
        self.num_parallel = max(1, num_parallel)
        self.root = None

        self.rule_name_to_id = {name: get_rule_id(name) for name in get_all_rule_names()}
        self.num_rules = get_num_rules()

    def _expand_node(self, node: Node, add_dirichlet: bool = False):
        """
        扩展节点：调用网络获取策略和价值，构建子节点候选列表。
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

        mask = torch.zeros(self.num_rules, dtype=torch.bool, device=self.device)
        for act in legal_acts:
            idx = self.rule_name_to_id.get(act.name, -1)
            if idx >= 0:
                mask[idx] = True

        try:
            norm_expr = parse_expr(srepr(node.state.expr))
        except Exception:
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
            combined = list(zip(node.sorted_actions, node.sorted_probs))
            combined.sort(key=lambda x: x[1], reverse=True)
            node.sorted_actions, node.sorted_probs = zip(*combined)
            node.sorted_actions = list(node.sorted_actions)
            node.sorted_probs = np.array(node.sorted_probs)

        node.is_expanded = True
        node.num_unlocked = 0

    def _progressive_expand(self, node: Node) -> bool:
        """
        渐进扩展：根据节点的访问次数动态增加子节点，返回是否创建了新子节点。
        """
        if not node.is_expanded:
            self._expand_node(node)
        if node.terminal:
            return False
        total = len(node.sorted_actions)
        if total == 0:
            return False
        # 动态计算允许扩展的最大子节点数
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
            # 执行动作得到后继状态
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

    def _select(self, node: Node) -> Tuple[Node, List[Tuple[Node, Action]]]:
        """
        选择阶段：从 node 开始，递归选择最优子节点直到叶节点。
        返回 (叶节点, 路径列表)
        """
        path = []
        visited_hashes = set()
        while True:
            if node.children:
                best_score = -float('inf')
                best_act = None
                best_child = None
                total_n = node.n
                # 探索常数可以随总模拟次数衰减
                c_puct_cur = self.c_puct
                for act, child in node.children.items():
                    # 标准 PUCT 公式，考虑虚拟损失后 child.n 可能包含虚拟计数
                    u = c_puct_cur * child.prior_p * math.sqrt(total_n + 1e-8) / (child.n + 1e-8)
                    # 使用 child.q 或 node.q 作为基准
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
            # 无子节点，尝试扩展
            if not node.is_expanded:
                self._expand_node(node)
            if node.terminal:
                break
            # 渐进扩展
            self._progressive_expand(node)
            if node.children:
                continue
            else:
                # 没有合法动作，标记为终止
                if node.num_unlocked >= len(node.sorted_actions):
                    node.terminal = True
                    node.terminal_reward = -0.1
                break
        return node, path

    def _simulate(self, add_virtual_loss: bool = True) -> None:
        """
        单次模拟，可选虚拟损失（用于并行调用）。
        """
        leaf, path = self._select(self.root)
        # 虚拟损失：在路径上添加临时惩罚
        if add_virtual_loss:
            for node, _ in path:
                node.add_virtual_loss()
            leaf.add_virtual_loss()

        # 评估叶节点
        if leaf.terminal:
            leaf_value = leaf.terminal_reward
        else:
            if not leaf.is_expanded:
                self._expand_node(leaf)
            leaf_value = leaf.value if hasattr(leaf, 'value') else 0.0

        # 反向传播
        leaf.update(leaf_value)
        accumulated = leaf_value
        for parent, act in reversed(path):
            child = parent.children[act]
            shaping = child.heuristic_reward
            value_to_parent = self.gamma * accumulated + shaping - self.step_penalty
            parent.update(value_to_parent)
            accumulated = value_to_parent

        # 恢复虚拟损失
        if add_virtual_loss:
            leaf.revert_virtual_loss(leaf_value)
            for node, act in reversed(path):
                child = node.children[act]
                node.revert_virtual_loss(child.q)   # 使用 child 的新 Q 值恢复

    def search(self, state: IntegrationState) -> Dict[Action, int]:
        """
        执行 MCTS 搜索，返回动作访问次数字典。
        """
        self.root = Node(state)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)

        # 探索常数调度（每次模拟后衰减）
        start_time = time.time()
        for sim in range(self.num_simulations):
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                break
            # 更新探索常数
            if self.c_puct_decay > 0:
                self.c_puct = self.base_c_puct * (1.0 - self.c_puct_decay) ** sim
            self._simulate(add_virtual_loss=False)   # 单线程时不需要虚拟损失

        return {act: child.n for act, child in self.root.children.items()}

    def parallel_search(self, state: IntegrationState) -> Dict[Action, int]:
        """
        并行 MCTS 搜索（多线程）。
        每个线程执行一次模拟，使用虚拟损失避免冲突。
        """
        self.root = Node(state)
        self._expand_node(self.root, add_dirichlet=True)
        self._progressive_expand(self.root)

        start_time = time.time()
        remaining = self.num_simulations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = []
            while remaining > 0:
                # 提交一批模拟任务
                batch = min(self.num_parallel, remaining)
                for _ in range(batch):
                    if self.timeout is not None and (time.time() - start_time) > self.timeout:
                        break
                    futures.append(executor.submit(self._simulate, add_virtual_loss=True))
                remaining -= batch
                # 可选：等待所有完成
                for f in futures:
                    f.result()
                # 清除已完成 futures
                futures = []
                if self.timeout is not None and (time.time() - start_time) > self.timeout:
                    break

        return {act: child.n for act, child in self.root.children.items()}

    def get_action_probs(self, state: IntegrationState, temperature: float = 1.0) -> Tuple[List[Action], List[float]]:
        """返回动作及其概率分布"""
        if self.num_parallel > 1:
            action_counts = self.parallel_search(state)
        else:
            action_counts = self.search(state)
        if not action_counts:
            return [], []
        actions = list(action_counts.keys())
        counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
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

    def get_trajectory(self, state: IntegrationState, temperature: float = 1.0,
                       include_metadata: bool = True) -> List[dict]:
        """
        生成完整轨迹，用于训练。
        """
        trajectory = []
        cur_state = state
        start_time = time.time()
        for step_idx in range(self.max_depth):
            if self.timeout is not None and (time.time() - start_time) > self.timeout:
                break
            if self.num_parallel > 1:
                action_counts = self.parallel_search(cur_state)
            else:
                action_counts = self.search(cur_state)
            if not action_counts:
                break

            # 温度退火
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

            # 根据访问次数采样动作
            actions = list(action_counts.keys())
            counts = np.array([action_counts[a] for a in actions], dtype=np.float64)
            probs = counts / counts.sum()
            action = np.random.choice(actions, p=probs)

            entry = {
                "state": cur_state,
                "policy_target": global_policy_target.copy(),
                "value_target": self.root.q if self.root else 0.0,
                "action": action,
                "q_values": {act.name: child.q for act, child in self.root.children.items()} if self.root and self.root.children else {}
            }
            if include_metadata:
                entry["rule_count"] = self.num_rules
            trajectory.append(entry)

            # 执行动作
            next_state, reward, done, info = self.env.step(cur_state, action)
            if done or next_state.depth >= self.max_depth:
                break
            cur_state = next_state
        return trajectory