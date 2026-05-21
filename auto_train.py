# auto_train.py
import os
import pickle
import random
import math
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp

# 统一顶部导入
from core.rules import RULE_DICT, RULE_NAMES, MathRuleBase
from core.engine import MCTS
from core.network import MathAlphaZeroNet
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from core.state import IntegrationState


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
    poly_coeff = sp.Poly(random.randint(1, 3) * x + random.randint(1, 2), x)
    return coeff * poly_coeff * base


def generate_random_problem(difficulty: str = "easy") -> sp.Expr:
    x = sp.Symbol('x')
    if difficulty == "easy":
        F = _generate_primitive_easy(x)
    elif difficulty == "medium":
        F = _generate_primitive_medium(x)
    else:  # hard
        F = _generate_primitive_hard(x)
    f = sp.diff(F, x)
    f = sp.simplify(f)
    return f


def normalize_expr(expr: sp.Expr) -> str:
    """将表达式规范化为字符串，用于去重和键值存储"""
    simplified = sp.simplify(expr)
    return str(simplified)


# ----------------------------- 2.1 优化版经验池管理 -----------------------------
class ExperienceBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.data_store = {}  # {问题规范化字符串: (步数, 轨迹条目列表)}
        self.buffer_list = []  # 存储所有 (state_tensor, policy_tensor, value_tensor)
        self.problem_to_indices = {}  # {问题规范化字符串: [索引列表]} 快速定位轨迹条目

    def push_trajectory(self, problem_expr_str, trajectory_data):
        new_steps = len(trajectory_data)
        norm_key = normalize_expr(sp.sympify(problem_expr_str))
        if norm_key in self.data_store:
            old_steps, _ = self.data_store[norm_key]
            if new_steps >= old_steps:
                return False
            else:
                print(f"🔥 [2.1 路径自我迭代] 💥 压缩步数: {old_steps} 步 -> {new_steps} 步！正在冲刷旧臃肿数据...")
                self._remove_trajectory(norm_key)

        # 记录新轨迹
        self.data_store[norm_key] = (new_steps, trajectory_data)
        start_idx = len(self.buffer_list)
        for item in trajectory_data:
            self.buffer_list.append(item)
        end_idx = len(self.buffer_list)
        self.problem_to_indices[norm_key] = list(range(start_idx, end_idx))

        # 容量限制：删除最旧的问题（按先进先出）
        while len(self.buffer_list) > self.capacity:
            first_key = next(iter(self.data_store.keys()))
            self._remove_trajectory(first_key)
        return True

    def _remove_trajectory(self, norm_key):
        if norm_key in self.data_store:
            _, old_entries = self.data_store[norm_key]
            indices_to_remove = self.problem_to_indices.get(norm_key, [])
            for idx in sorted(indices_to_remove, reverse=True):
                if 0 <= idx < len(self.buffer_list):
                    self.buffer_list.pop(idx)
            del self.data_store[norm_key]
            if norm_key in self.problem_to_indices:
                del self.problem_to_indices[norm_key]
            self._rebuild_indices()

    def _rebuild_indices(self):
        """重建问题到索引列表的映射（在删除后调用）"""
        new_mapping = {}
        for key, (_, entries) in self.data_store.items():
            indices = []
            for entry in entries:
                try:
                    idx = self.buffer_list.index(entry)
                    indices.append(idx)
                except ValueError:
                    pass
            if indices:
                new_mapping[key] = indices
        self.problem_to_indices = new_mapping

    def sample(self, batch_size: int):
        return random.sample(self.buffer_list, batch_size)

    def __len__(self):
        return len(self.buffer_list)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self.data_store, self.buffer_list, self.problem_to_indices), f)

    def load(self, path: str):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'rb') as f:
                saved_data = pickle.load(f)
                if isinstance(saved_data, tuple) and len(saved_data) == 3:
                    self.data_store, self.buffer_list, self.problem_to_indices = saved_data
                else:
                    self.buffer_list = saved_data
                    self.data_store = {}
                    self.problem_to_indices = {}
                    self._rebuild_indices()
                print(f"✅ 2.1 经验池激活，当前包含 {len(self.buffer_list)} 条高价值去重记录，覆盖 {len(self.data_store)} 道核心题型")
        else:
            print("ℹ️ 经验池为空或不存在，创建新经验池")


# ----------------------------- 训练主循环 -----------------------------
def main():
    print("====== MathAlphaZero 2.1 效率驱动型进化系统启动 ======")

    # 超参数配置
    MAX_SIMULATIONS = 100
    NUM_EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SAVE_INTERVAL = 10
    DECAY_FACTOR = 0.92
    PROBLEM_TIMEOUT = 60.0

    preprocessor = MathPreprocessor(max_len=128)
    rules = MathRuleBase()
    validator = MathValidator()

    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=rules.num_actions,
        d_model=128, nhead=4, num_layers=3
    )
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/brain.pth"):
        net.load_state_dict(torch.load("data/brain.pth"))
        print("✅ 加载已有大脑权重 (继承历史记忆)")

    memory = ExperienceBuffer(capacity=20000)
    memory.load("data/memory.pkl")

    total_games = 0
    current_run_successes = 0  # 记录当前运行成功解题的次数
    solved_history = set(normalize_expr(sp.sympify(key)) for key in memory.data_store.keys())

    # 记录全局训练时间
    global_start_time = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- 世代 {epoch}/{NUM_EPOCHS} ---")

        if epoch <= 40:
            DIFFICULTY = "easy"
        elif epoch <= 120:
            DIFFICULTY = "medium"
        else:
            DIFFICULTY = "hard"

        # 生成未解决过的新题目
        for _ in range(50):
            expr = generate_random_problem(DIFFICULTY)
            norm_expr = normalize_expr(expr)
            if norm_expr not in solved_history:
                break
        else:
            print(f"⚠️ {DIFFICULTY} 难度的旧题型已刷完，复盘历史题目以优化步数。")
            expr = generate_random_problem(DIFFICULTY)
            norm_expr = normalize_expr(expr)

        print(f"📝 探索题目 [{DIFFICULTY}]: ∫ {expr} dx")
        total_games += 1

        prob_start_time = time.perf_counter()
        x = sp.Symbol('x')
        init_state = IntegrationState(expr=sp.Integral(expr, x))

        mcts = MCTS(network=net, preprocessor=preprocessor, num_simulations=MAX_SIMULATIONS, timeout=PROBLEM_TIMEOUT)

        trajectory = mcts.get_trajectory(init_state, temperature=1.0)

        elapsed_time = time.perf_counter() - prob_start_time
        if elapsed_time > PROBLEM_TIMEOUT:
            print(f"⏱️  【单题强杀】该题搜索耗时 {elapsed_time:.1f} 秒，超过1分钟限制！直接切入下一道题...")
            continue

        success = False
        path = []
        final_expr = expr

        if trajectory:
            last_step = trajectory[-1]
            next_state_raw, reward, done, info = mcts.env.step(last_step["state"], last_step["action"])
            for step in trajectory:
                path.append((step["state"].expr, step["action"].name))
            if done and reward > 0:
                success = True
                final_expr = next_state_raw.expr

        if success and path:
            if not validator.verify_integral(expr, final_expr):
                success = False

        if success:
            current_run_successes += 1
            expr_str = str(expr)
            total_steps = len(trajectory)
            print(f"✅ 解题成功！实际推导步数: {total_steps}")

            trajectory_entries = []
            for idx, step_data in enumerate(trajectory):
                state_tensor = preprocessor.state_to_tensor(step_data["state"].expr)
                policy_target = step_data["policy_target"]
                remaining_steps = total_steps - idx
                discounted_value = 1.0 * (DECAY_FACTOR ** remaining_steps)

                state_cpu = state_tensor.cpu().clone()
                policy_cpu = torch.tensor(policy_target, dtype=torch.float32)
                value_cpu = torch.tensor([discounted_value], dtype=torch.float32)
                trajectory_entries.append((state_cpu, policy_cpu, value_cpu))

            is_new_or_better = memory.push_trajectory(expr_str, trajectory_entries)
            if is_new_or_better:
                norm_key = normalize_expr(expr)
                if norm_key not in solved_history:
                    solved_history.add(norm_key)
                print(f"📚 经验池容量更新: {len(memory)}")
        else:
            print("❌ 未找到有效解")

        if time.perf_counter() - prob_start_time > PROBLEM_TIMEOUT:
            print(f"⏱️  【后期强杀】后期验证耗时过长，直接切入下一道题...")
            continue

        # 神经网络训练更新
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            batch_states = torch.cat([item[0] for item in batch], dim=0)
            batch_policies = torch.stack([item[1] for item in batch])
            batch_values = torch.stack([item[2] for item in batch])

            policy_logits, pred_values = net(batch_states)
            log_probs = nn.LogSoftmax(dim=1)(policy_logits)
            policy_loss = - (batch_policies * log_probs).sum(dim=1).mean()
            value_loss = nn.MSELoss()(pred_values, batch_values)
            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"🧠 训练更新: Loss = {total_loss.item():.4f}")

        if epoch % SAVE_INTERVAL == 0:
            torch.save(net.state_dict(), "data/brain.pth")
            memory.save("data/memory.pkl")
            print(f"💾 2.1 进度已存档 (世代 {epoch})")

    # ==================== 训练结束：统计与对比 ====================
    torch.save(net.state_dict(), "data/brain_final.pth")
    memory.save("data/memory_final.pkl")

    global_end_time = time.perf_counter()
    total_elapsed_time = global_end_time - global_start_time
    avg_time_per_problem = total_elapsed_time / total_games if total_games > 0 else 0.0
    current_accuracy = (current_run_successes / total_games * 100) if total_games > 0 else 0.0

    print("\n" + "=" * 60)
    print("🎉 200 个世代训练全部完成！")
    print(f"✅ 总题数: {total_games} 题 | 成功解出: {current_run_successes} 题")
    print(f"🎯 训练期解题准确率: {current_accuracy:.2f}%")
    print(f"⏱️ 训练总耗时: {total_elapsed_time:.1f} 秒 (平均单题流转耗时: {avg_time_per_problem:.2f} 秒)")

    # 能力提升历史对比逻辑
    history_path = "data/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        best_acc = history.get("best_accuracy", 0.0)
        best_time = history.get("best_avg_time", float('inf'))

        print("\n📈 ====== 能力提升历史对比 ======")

        # 对比准确率
        if current_accuracy > best_acc:
            print(
                f"🚀 【准确率突破】 创造历史最佳！({best_acc:.2f}% -> {current_accuracy:.2f}%) 提升了 {current_accuracy - best_acc:.2f}%")
            history["best_accuracy"] = current_accuracy
        else:
            print(f"📊 【准确率维稳】 当前 {current_accuracy:.2f}% (历史最佳为 {best_acc:.2f}%)")

        # 对比解题效率 (越低越好)
        if avg_time_per_problem < best_time:
            print(
                f"⚡ 【速度突破】 推导与学习效率变快！(平均单题 {best_time:.2f} 秒 -> {avg_time_per_problem:.2f} 秒) 缩短了 {best_time - avg_time_per_problem:.2f} 秒")
            history["best_avg_time"] = avg_time_per_problem
        else:
            print(f"🐢 【速度维稳】 当前单题耗时 {avg_time_per_problem:.2f} 秒 (历史最佳为 {best_time:.2f} 秒)")

        # 记录本次数据
        history["last_accuracy"] = current_accuracy
        history["last_avg_time"] = avg_time_per_problem
    else:
        print("\n📈 这是系统的第一次完整训练，已记录初始基准数据，将在下一次训练时进行提升对比！")
        history = {
            "best_accuracy": current_accuracy,
            "best_avg_time": avg_time_per_problem,
            "last_accuracy": current_accuracy,
            "last_avg_time": avg_time_per_problem
        }

    # 保存历史记录
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    print("=============================================================\n")


if __name__ == "__main__":
    main()