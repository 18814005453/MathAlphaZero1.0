# auto_train.py
import os
import pickle
import random
import time
import json
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp
from collections import defaultdict, deque

# 统一导入
import knowledge.rules
from knowledge.rule_registry import get_all_rule_names, get_num_rules, build_action_space
from core.engine import MCTS
from core.network import MathAlphaZeroNet
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from core.state import IntegrationState
from core.env import IntegrationEnv

# ==================== 逆向生成法（保证可积性） ====================
def _random_coefficient():
    if random.random() < 0.7:
        return random.randint(1, 5) * random.choice([-1, 1])
    else:
        num = random.randint(1, 3)
        den = random.randint(2, 4)
        return sp.Rational(num, den) * random.choice([-1, 1])

def _generate_primitive_easy(x):
    choices = [
        x ** n for n in range(1, 4)
    ] + [
        sp.sin(k * x) for k in range(1, 3)
    ] + [
        sp.cos(k * x) for k in range(1, 3)
    ] + [
        sp.exp(k * x) for k in range(1, 3)
    ] + [
        sp.exp(-x)
    ]
    base = random.choice(choices)
    coeff = _random_coefficient()
    return coeff * base

def _generate_primitive_medium(x):
    prod_choices = [
        x * sp.sin(x), x * sp.cos(x), x * sp.exp(x),
        x ** 2 * sp.exp(x), x * sp.sin(2 * x)
    ]
    comp_choices = [
        sp.sin(x ** 2), sp.exp(sp.sin(x)), sp.log(x + 2), sp.atan(x)
    ]
    base = random.choice(prod_choices + comp_choices)
    coeff = _random_coefficient()
    return coeff * base

def _generate_primitive_hard(x):
    rational = [
        1 / (x ** 2 + 1), x / (x ** 2 + 1), 1 / ((x + 1) ** 2), sp.log(x ** 2 + 1)
    ]
    nested = [
        sp.exp(x ** 2), sp.sin(x ** 2), sp.cos(x ** 2), sp.exp(sp.sin(x))
    ]
    mixed = [
        x * sp.atan(x), x * sp.log(x + 1)
    ]
    base = random.choice(rational + nested + mixed)
    coeff = _random_coefficient()
    poly_coeff = random.randint(1, 3) * x + random.randint(1, 2)
    return coeff * poly_coeff * base

def generate_random_problem(difficulty: str = "easy") -> sp.Expr:
    x = sp.Symbol('x')
    if difficulty == "easy":
        F = _generate_primitive_easy(x)
    elif difficulty == "medium":
        F = _generate_primitive_medium(x)
    else:
        F = _generate_primitive_hard(x)
    f = sp.diff(F, x)
    f = sp.simplify(f)
    return f

def normalize_expr(expr: sp.Expr) -> str:
    simplified = sp.simplify(expr)
    return str(simplified)

def ast_complexity(expr):
    """递归计算 AST 节点总数"""
    if expr.is_Atom:
        return 1
    return 1 + sum(ast_complexity(arg) for arg in expr.args)

def ast_depth(expr):
    """AST 最大深度（难度评估）"""
    if expr.is_Atom:
        return 1
    return 1 + max(ast_depth(arg) for arg in expr.args)

def generate_problem_with_ast_depth(max_depth: int, max_attempts=30) -> sp.Expr:
    """生成 AST 深度不超过 max_depth 的积分题目"""
    for _ in range(max_attempts):
        # 随机难度，但倾向于生成较高难度的再剪枝
        diff = random.choices(["easy", "medium", "hard"], weights=[0.2, 0.3, 0.5])[0]
        expr = generate_random_problem(diff)
        if ast_depth(expr) <= max_depth:
            return expr
    # 保底：生成一个简单题
    return generate_random_problem("easy")

# ----------------------------- 经验池管理（支持成功轨迹优先） -----------------------------
class ExperienceBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.data_store = {}       # {问题规范化字符串: (步数, 轨迹条目列表, success)}
        self.buffer_list = []      # 扁平存储 (state_tensor, policy_tensor, value_tensor, action_id, mask_tensor)
        self.problem_to_indices = {}  # {问题规范化字符串: [索引列表]}

    def push_trajectory(self, problem_expr_str, trajectory_data, success: bool):
        norm_key = normalize_expr(sp.sympify(problem_expr_str))
        new_steps = len(trajectory_data)
        if norm_key in self.data_store:
            old_steps, old_entries, old_success = self.data_store[norm_key]
            if old_success and not success:
                return False
            if not old_success and success:
                self._remove_trajectory(norm_key)
            elif success == old_success and new_steps >= old_steps:
                return False
            else:
                self._remove_trajectory(norm_key)
        self.data_store[norm_key] = (new_steps, trajectory_data, success)
        start_idx = len(self.buffer_list)
        for item in trajectory_data:
            self.buffer_list.append(item)
        end_idx = len(self.buffer_list)
        self.problem_to_indices[norm_key] = list(range(start_idx, end_idx))
        while len(self.buffer_list) > self.capacity:
            first_key = next(iter(self.data_store.keys()))
            self._remove_trajectory(first_key)
        return True

    def _remove_trajectory(self, norm_key):
        if norm_key in self.data_store:
            _, old_entries, _ = self.data_store[norm_key]
            indices_to_remove = self.problem_to_indices.get(norm_key, [])
            for idx in sorted(indices_to_remove, reverse=True):
                if 0 <= idx < len(self.buffer_list):
                    self.buffer_list.pop(idx)
            del self.data_store[norm_key]
            if norm_key in self.problem_to_indices:
                del self.problem_to_indices[norm_key]
            self._rebuild_indices()

    def _rebuild_indices(self):
        id_to_idx = {id(item): idx for idx, item in enumerate(self.buffer_list)}
        new_mapping = {}
        for key, (_, entries, _) in self.data_store.items():
            indices = [id_to_idx.get(id(entry)) for entry in entries if id_to_idx.get(id(entry)) is not None]
            if indices:
                new_mapping[key] = indices
        self.problem_to_indices = new_mapping

    def sample(self, batch_size: int):
        return random.sample(self.buffer_list, min(batch_size, len(self.buffer_list)))

    def __len__(self):
        return len(self.buffer_list)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self.data_store, self.buffer_list, self.problem_to_indices), f)
        # 额外保存 pattern_miner 需要的字典格式（仅成功轨迹）
        miner_data = {"actions": [], "complexities": [], "reward": [], "policy_probs": [], "q_values": []}
        for norm_key, (steps, entries, success) in self.data_store.items():
            if not success:
                continue
            actions_seq = [entry[3] for entry in entries]
            miner_data["actions"].append(actions_seq)
            miner_data["complexities"].append(steps)
            miner_data["reward"].append(1.0)
            # 为了价值落差挖掘，记录策略概率和 Q 值（如果有）
            policy_probs_seq = []
            q_vals_seq = []
            for entry in entries:
                if len(entry) >= 6:
                    policy_probs_seq.append(entry[5])  # policy_probs
                    q_vals_seq.append(entry[6])       # q_value
                else:
                    policy_probs_seq.append(0.0)
                    q_vals_seq.append(0.0)
            miner_data["policy_probs"].append(policy_probs_seq)
            miner_data["q_values"].append(q_vals_seq)
        miner_path = path.replace(".pkl", "_for_miner.pkl")
        with open(miner_path, 'wb') as f_miner:
            pickle.dump(miner_data, f_miner)
        print(f"✅ 已同步生成 pattern_miner 兼容数据: {miner_path}")

    def load(self, path: str):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                if isinstance(saved_data, tuple) and len(saved_data) == 3:
                    self.data_store, self.buffer_list, self.problem_to_indices = saved_data
                    for key, value in list(self.data_store.items()):
                        if len(value) == 2:
                            steps, entries = value
                            self.data_store[key] = (steps, entries, True)
                    if self.buffer_list and len(self.buffer_list[0]) == 4:
                        print("⚠️ 检测到旧格式经验池，清空重新开始。")
                        self.data_store.clear()
                        self.buffer_list.clear()
                        self.problem_to_indices.clear()
                else:
                    self.buffer_list = saved_data
                    self.data_store = {}
                    self.problem_to_indices = {}
                    self._rebuild_indices()
                print(f"✅ 经验池加载完成，共 {len(self.buffer_list)} 条记录，覆盖 {len(self.data_store)} 道题目")
        else:
            print("ℹ️ 经验池为空，创建新经验池")

# ----------------------------- 动态课程学习 -----------------------------
class CurriculumTracker:
    def __init__(self, window_size=50):
        self.depth_performance = defaultdict(lambda: deque(maxlen=window_size))
        self.current_max_depth = 2

    def update(self, depth: int, success: bool):
        self.depth_performance[depth].append(1.0 if success else 0.0)

    def get_current_depth(self, threshold=0.7) -> int:
        for d in sorted(self.depth_performance.keys()):
            perf = self.depth_performance[d]
            if len(perf) > 0 and sum(perf) / len(perf) >= threshold:
                self.current_max_depth = max(self.current_max_depth, d + 1)
        return self.current_max_depth

# ----------------------------- Pending Buffer（高价值失败轨迹） -----------------------------
class PendingBuffer:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.problems = []  # (expr, partial_traj, final_complexity, initial_complexity)

    def add(self, expr, partial_traj, final_complexity, initial_complexity):
        if final_complexity < 0.8 * initial_complexity:
            self.problems.append((expr, partial_traj, final_complexity, initial_complexity))
            if len(self.problems) > self.capacity:
                self.problems.pop(0)

    def retry_all(self, net, preprocessor, memory, max_simulations=800, timeout=120.0):
        successes = 0
        for expr, _, final_c, init_c in self.problems:
            print(f"🔄 重试 pending 问题: ∫ {expr} dx (化简程度 {1-final_c/init_c:.2%})")
            mcts = MCTS(network=net, preprocessor=preprocessor,
                        num_simulations=max_simulations, timeout=timeout, max_depth=40)
            state = IntegrationState(sp.Integral(expr, sp.Symbol('x')))
            traj = mcts.get_trajectory(state)
            if traj:
                last_step = traj[-1]
                env = IntegrationEnv(max_steps=40)
                next_state, reward, done, _ = env.step(last_step["state"], last_step["action"])
                if done and reward > 0.8:
                    # 成功，将轨迹存入 memory
                    # 转换 trajectory 格式
                    trajectory_entries = []
                    for step_data in traj:
                        state_tensor = preprocessor.state_to_tensor(step_data["state"].expr)
                        policy_target = step_data["policy_target"]
                        action_id = step_data["action"].id
                        mask_tensor = torch.ones(get_num_rules(), dtype=torch.bool)  # 简化，实际应计算合法动作掩码
                        trajectory_entries.append((state_tensor.cpu(), torch.tensor(policy_target, dtype=torch.float32),
                                                   torch.tensor([reward], dtype=torch.float32), action_id, mask_tensor))
                    memory.push_trajectory(str(expr), trajectory_entries, True)
                    successes += 1
        print(f"✅ Pending 重试完成，成功 {successes}/{len(self.problems)} 道")
        self.problems.clear()
        return successes

# ----------------------------- 主训练循环 -----------------------------
def main():
    print("====== MathAlphaZero 2.2 动态课程学习 + 价值落差挖掘系统启动 ======")

    # 超参数
    MAX_SIMULATIONS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DECAY_FACTOR = 0.92
    PROBLEM_TIMEOUT = 40.0
    TRAIN_ITERATIONS = 20
    EPOCHS = 100
    PROBLEMS_PER_EPOCH = 50

    preprocessor = MathPreprocessor(max_len=128)
    validator = MathValidator()

    # 确保规则注册表已构建
    build_action_space()
    rule_names = get_all_rule_names()
    num_actions = get_num_rules()
    print(f"📚 规则动作空间: {num_actions} 个原子规则 + 待生成宏规则")

    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=num_actions,
        d_model=128, nhead=4, num_layers=3
    )
    # 建立规则 ID 映射（与注册表一致）
    action_ids = list(range(num_actions))
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids=action_ids)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/brain.pth"):
        net.load_state_dict(torch.load("data/brain.pth"))
        print("✅ 加载已有大脑权重")

    memory = ExperienceBuffer(capacity=20000)
    memory.load("data/memory_final.pkl")

    curriculum = CurriculumTracker(window_size=30)
    pending = PendingBuffer(capacity=100)

    # 已成功题目集合（防止重复生成）
    solved_history = set()
    for key, (_, _, success) in memory.data_store.items():
        if success:
            solved_history.add(normalize_expr(sp.sympify(key)))

    total_games = 0
    epoch_successes = 0

    for epoch in range(EPOCHS):
        current_depth = curriculum.get_current_depth(threshold=0.65)
        print(f"\n{'='*60}\n🔥 Epoch {epoch+1}/{EPOCHS} | 当前课程深度: {current_depth} | 总成功题目数: {len(solved_history)}")
        epoch_successes = 0

        # 生成题目
        problem_list = []
        for _ in range(PROBLEMS_PER_EPOCH):
            expr = generate_problem_with_ast_depth(current_depth, max_attempts=20)
            norm = normalize_expr(expr)
            # 去重：如果已经成功过，跳过（但允许重新尝试失败过的题目）
            if norm in solved_history:
                continue
            problem_list.append((expr, ast_depth(expr)))

        if not problem_list:
            print("⚠️ 没有新题目可生成，尝试增加深度或放宽去重")
            continue

        print(f"📋 本次将求解 {len(problem_list)} 道题目")

        for expr, depth in problem_list:
            total_games += 1
            print(f"\n📝 求解: ∫ {expr} dx (深度 {depth})")
            state = IntegrationState(sp.Integral(expr, sp.Symbol('x')))
            mcts = MCTS(
                network=net,
                preprocessor=preprocessor,
                num_simulations=MAX_SIMULATIONS,
                timeout=PROBLEM_TIMEOUT,
                max_depth=30
            )
            trajectory = mcts.get_trajectory(state, temperature=1.0)

            if not trajectory:
                print("❌ 无轨迹生成")
                curriculum.update(depth, False)
                continue

            # 获取最终状态
            last_step = trajectory[-1]
            env = IntegrationEnv(max_steps=30, time_limit=PROBLEM_TIMEOUT)
            next_state, reward, done, info = env.step(last_step["state"], last_step["action"])
            success = (done and reward > 0.8 and validator.verify_integral(expr, next_state.expr))

            if success:
                epoch_successes += 1
                solved_history.add(normalize_expr(expr))
                print(f"✅ 解题成功！步数: {len(trajectory)}, 最终奖励: {reward:.3f}")
                curriculum.update(depth, True)
                # 转换为经验池格式
                trajectory_entries = []
                total_steps = len(trajectory)
                for idx_step, step_data in enumerate(trajectory):
                    state_tensor = preprocessor.state_to_tensor(step_data["state"].expr)
                    policy_target = step_data["policy_target"]
                    action_id = step_data["action"].id
                    remaining_steps = total_steps - idx_step
                    discounted_value = reward * (DECAY_FACTOR ** remaining_steps)
                    # 掩码（简化：全1）
                    mask_tensor = torch.ones(num_actions, dtype=torch.bool)
                    trajectory_entries.append((
                        state_tensor.cpu().clone(),
                        torch.tensor(policy_target, dtype=torch.float32),
                        torch.tensor([discounted_value], dtype=torch.float32),
                        action_id,
                        mask_tensor,
                        step_data.get("policy_target", [0])[action_id] if "policy_target" in step_data else 0.0,
                        step_data.get("value_target", 0.0)
                    ))
                memory.push_trajectory(str(expr), trajectory_entries, True)
            else:
                print(f"❌ 解题失败 (最终奖励 {reward:.3f})")
                curriculum.update(depth, False)
                # 若化简较多，加入 pending
                init_c = state.ast_complexity()
                final_c = next_state.ast_complexity() if 'next_state' in locals() else init_c
                if final_c < 0.8 * init_c:
                    pending.add(expr, trajectory, final_c, init_c)

        # 每10个epoch重试 pending 问题
        if epoch % 10 == 0 and len(pending.problems) > 0:
            print(f"\n🔄 尝试重试 {len(pending.problems)} 个待解决题目...")
            pending.retry_all(net, preprocessor, memory, max_simulations=800, timeout=120.0)

        # 训练网络
        if len(memory) >= BATCH_SIZE:
            print(f"\n🧠 训练神经网络，经验池大小: {len(memory)}")
            device = next(net.parameters()).device
            for train_iter in range(TRAIN_ITERATIONS):
                batch = memory.sample(BATCH_SIZE)
                batch_states, batch_policies, batch_values, batch_masks = [], [], [], []
                for item in batch:
                    state_tensor, policy_tensor, value_tensor, _, mask_tensor = item[:5]
                    batch_states.append(state_tensor)
                    # 动态调整策略向量长度
                    if policy_tensor.size(0) < num_actions:
                        pad = torch.zeros(num_actions - policy_tensor.size(0), dtype=policy_tensor.dtype)
                        policy_tensor = torch.cat([policy_tensor, pad])
                    elif policy_tensor.size(0) > num_actions:
                        policy_tensor = policy_tensor[:num_actions]
                    batch_policies.append(policy_tensor)
                    batch_values.append(value_tensor)
                    if mask_tensor.size(0) < num_actions:
                        pad = torch.zeros(num_actions - mask_tensor.size(0), dtype=torch.bool)
                        mask_tensor = torch.cat([mask_tensor, pad])
                    elif mask_tensor.size(0) > num_actions:
                        mask_tensor = mask_tensor[:num_actions]
                    batch_masks.append(mask_tensor)
                batch_states = torch.cat(batch_states, dim=0).to(device)
                batch_policies = torch.stack(batch_policies).to(device)
                batch_values = torch.stack(batch_values).to(device)
                batch_masks = torch.stack(batch_masks).to(device)

                # 策略标签归一化
                batch_policies = batch_policies.masked_fill(~batch_masks, 0.0)
                row_sums = batch_policies.sum(dim=1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                batch_policies = batch_policies / (row_sums + 1e-8)

                policy_logits, pred_values = net(batch_states, batch_masks)
                log_probs = nn.LogSoftmax(dim=1)(policy_logits)
                policy_loss = - (batch_policies * log_probs).sum(dim=1).mean()
                value_loss = nn.MSELoss()(pred_values, batch_values)
                total_loss = policy_loss + value_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                if (train_iter+1) % 5 == 0:
                    print(f"  训练迭代 {train_iter+1}/{TRAIN_ITERATIONS}: Loss = {total_loss.item():.4f}")

        # 保存检查点
        torch.save(net.state_dict(), "data/brain_final.pth")
        memory.save("data/memory_final.pkl")
        epoch_accuracy = epoch_successes / max(1, len(problem_list)) * 100
        print(f"📊 Epoch {epoch+1} 总结: 解题成功 {epoch_successes}/{len(problem_list)} ({epoch_accuracy:.1f}%)")

        # 记录历史准确率
        history_path = "data/training_history.json"
        history = {}
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        history.setdefault("accuracy_per_epoch", []).append(epoch_accuracy)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

    print("\n🎉 训练完成！")

if __name__ == "__main__":
    main()