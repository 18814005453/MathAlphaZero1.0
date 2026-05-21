import os
import pickle
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp

# 统一顶部导入
from knowledge.rules import MathRuleBase, RULE_NAMES   # 引入 RULE_NAMES
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
        self.buffer_list = []  # 存储所有 (state_tensor, policy_tensor, value_tensor, action_ids)
        self.problem_to_indices = {}  # {问题规范化字符串: [索引列表]} 快速定位轨迹条目

    def push_trajectory(self, problem_expr_str, trajectory_data):
        """
        trajectory_data: list of (state_tensor, policy_tensor, value_tensor, action_id)
        """
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
        # 原有训练数据保存格式（三元组 + 动作序列）
        with open(path, 'wb') as f:
            pickle.dump((self.data_store, self.buffer_list, self.problem_to_indices), f)

        # ===== 额外保存 pattern_miner 需要的字典格式 =====
        miner_data = {
            "actions": [],
            "complexities": [],
            "reward": []
        }
        for norm_key, (steps, entries) in self.data_store.items():
            actions_seq = [entry[3] for entry in entries]   # 从存储的条目中取出 action_id
            miner_data["actions"].append(actions_seq)
            miner_data["complexities"].append(steps)
            miner_data["reward"].append(1.0)   # 成功解题 reward 为 1
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
                else:
                    # 兼容旧版本只有 buffer_list 的情况
                    self.buffer_list = saved_data
                    self.data_store = {}
                    self.problem_to_indices = {}
                    self._rebuild_indices()
                print(f"✅ 2.1 经验池激活，当前包含 {len(self.buffer_list)} 条高价值去重记录，覆盖 {len(self.data_store)} 道核心题型")
        else:
            print("ℹ️ 经验池为空或不存在，创建新经验池")


# ----------------------------- 训练主循环（一次训练50题：10简单+25中等+15困难） -----------------------------
def main():
    print("====== MathAlphaZero 2.1 效率驱动型进化系统启动 ======")
    print("训练配置: 一次性生成50道题目 (10简单, 25中等, 15困难)")

    # 超参数配置
    MAX_SIMULATIONS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DECAY_FACTOR = 0.92
    PROBLEM_TIMEOUT = 60.0
    # 训练迭代次数（从经验池中采样训练的轮数）
    TRAIN_ITERATIONS = 10   # 50题完成后，进行10次梯度更新（每次一个batch）

    preprocessor = MathPreprocessor(max_len=128)
    rules = MathRuleBase()
    validator = MathValidator()

    # 初始化网络
    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=rules.num_actions,
        d_model=128, nhead=4, num_layers=3
    )
    net.refresh_rule_cache(
        RULE_NAMES,
        tokenizer_fn=preprocessor.state_to_tensor
    )

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/brain.pth"):
        net.load_state_dict(torch.load("data/brain.pth"))
        print("✅ 加载已有大脑权重 (继承历史记忆)")

    memory = ExperienceBuffer(capacity=20000)
    # 第一处修改：统一指向 memory_final.pkl
    memory.load("data/memory_final.pkl")

    # 已解决题目集合（用于去重）
    solved_history = set(normalize_expr(sp.sympify(key)) for key in memory.data_store.keys())

    total_games = 0
    current_run_successes = 0

    # ========== 构建题目列表 (10 easy + 25 medium + 15 hard) ==========
    problem_list = []
    # 简单题 10 道
    for _ in range(10):
        for _ in range(20):  # 最多尝试20次生成新题
            expr = generate_random_problem("easy")
            norm_expr = normalize_expr(expr)
            if norm_expr not in solved_history:
                problem_list.append((expr, "easy"))
                solved_history.add(norm_expr)
                break
        else:
            # 如果无法生成全新题，就接受重复（但记录警告）
            expr = generate_random_problem("easy")
            problem_list.append((expr, "easy"))
            print(f"⚠️ 简单题生成重复，接受题目: {expr}")
    # 中等题 25 道
    for _ in range(25):
        for _ in range(20):
            expr = generate_random_problem("medium")
            norm_expr = normalize_expr(expr)
            if norm_expr not in solved_history:
                problem_list.append((expr, "medium"))
                solved_history.add(norm_expr)
                break
        else:
            expr = generate_random_problem("medium")
            problem_list.append((expr, "medium"))
            print(f"⚠️ 中等题生成重复，接受题目: {expr}")
    # 困难题 15 道
    for _ in range(15):
        for _ in range(20):
            expr = generate_random_problem("hard")
            norm_expr = normalize_expr(expr)
            if norm_expr not in solved_history:
                problem_list.append((expr, "hard"))
                solved_history.add(norm_expr)
                break
        else:
            expr = generate_random_problem("hard")
            problem_list.append((expr, "hard"))
            print(f"⚠️ 困难题生成重复，接受题目: {expr}")

    print(f"\n📋 共计生成 {len(problem_list)} 道题目（简单:10, 中等:25, 困难:15）")
    global_start_time = time.perf_counter()

    # ========== 对每道题进行MCTS搜索，收集轨迹 ==========
    for idx, (expr, difficulty) in enumerate(problem_list, 1):
        print(f"\n📝 探索题目 {idx}/{len(problem_list)} [{difficulty}]: ∫ {expr} dx")
        total_games += 1

        prob_start_time = time.perf_counter()
        x = sp.Symbol('x')
        init_state = IntegrationState(expr=sp.Integral(expr, x))

        # 每次搜索前刷新规则缓存
        net.refresh_rule_cache(
            RULE_NAMES,
            tokenizer_fn=preprocessor.state_to_tensor
        )

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
            for idx_step, step_data in enumerate(trajectory):
                state_tensor = preprocessor.state_to_tensor(step_data["state"].expr)
                policy_target = step_data["policy_target"]
                action_id = step_data["action"].id
                remaining_steps = total_steps - idx_step
                discounted_value = 1.0 * (DECAY_FACTOR ** remaining_steps)

                state_cpu = state_tensor.cpu().clone()
                policy_cpu = torch.tensor(policy_target, dtype=torch.float32)
                value_cpu = torch.tensor([discounted_value], dtype=torch.float32)
                trajectory_entries.append((state_cpu, policy_cpu, value_cpu, action_id))

            is_new_or_better = memory.push_trajectory(expr_str, trajectory_entries)
            if is_new_or_better:
                print(f"📚 经验池容量更新: {len(memory)}")
        else:
            print("❌ 未找到有效解")

        if time.perf_counter() - prob_start_time > PROBLEM_TIMEOUT:
            print(f"⏱️  【后期强杀】后期验证耗时过长，直接切入下一道题...")
            continue

    # ========== 所有题目处理完毕，进行神经网络训练 ==========
    print("\n🧠 开始基于经验池训练神经网络...")
    if len(memory) >= BATCH_SIZE:
        for train_iter in range(TRAIN_ITERATIONS):
            batch = memory.sample(BATCH_SIZE)
            current_num_actions = net.get_rule_embeddings().size(0)
            batch_states = []
            batch_policies = []
            batch_values = []
            for item in batch:
                state, policy, value, _ = item
                batch_states.append(state)
                if policy.size(0) < current_num_actions:
                    pad = torch.zeros(current_num_actions - policy.size(0), dtype=policy.dtype)
                    policy = torch.cat([policy, pad])
                elif policy.size(0) > current_num_actions:
                    policy = policy[:current_num_actions]
                batch_policies.append(policy)
                batch_values.append(value)

            batch_states = torch.cat(batch_states, dim=0)
            batch_policies = torch.stack(batch_policies)
            batch_values = torch.stack(batch_values)

            net.refresh_rule_cache(
                RULE_NAMES,
                tokenizer_fn=preprocessor.state_to_tensor
            )
            policy_logits, pred_values = net(batch_states)

            log_probs = nn.LogSoftmax(dim=1)(policy_logits)
            policy_loss = - (batch_policies * log_probs).sum(dim=1).mean()
            value_loss = nn.MSELoss()(pred_values, batch_values)
            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"🧠 训练迭代 {train_iter+1}/{TRAIN_ITERATIONS}: Loss = {total_loss.item():.4f}")
    else:
        print(f"⚠️ 经验池数据不足（{len(memory)} < {BATCH_SIZE}），跳过训练")

    # 保存最终模型和经验池
    torch.save(net.state_dict(), "data/brain_final.pth")
    # 第三处（以及第二处如果存在）均指向 memory_final.pkl
    memory.save("data/memory_final.pkl")
    print("💾 最终模型和经验池已保存")

    global_end_time = time.perf_counter()
    total_elapsed_time = global_end_time - global_start_time
    avg_time_per_problem = total_elapsed_time / total_games if total_games > 0 else 0.0
    current_accuracy = (current_run_successes / total_games * 100) if total_games > 0 else 0.0

    print("\n" + "=" * 60)
    print("🎉 单次50题训练完成！")
    print(f"✅ 总题数: {total_games} 题 | 成功解出: {current_run_successes} 题")
    print(f"🎯 解题准确率: {current_accuracy:.2f}%")
    print(f"⏱️ 总耗时: {total_elapsed_time:.1f} 秒 (平均单题流转耗时: {avg_time_per_problem:.2f} 秒)")

    # 能力提升历史对比逻辑（保留与之前一致的统计）
    history_path = "data/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        best_acc = history.get("best_accuracy", 0.0)
        best_time = history.get("best_avg_time", float('inf'))

        print("\n📈 ====== 能力提升历史对比 ======")

        if current_accuracy > best_acc:
            print(f"🚀 【准确率突破】 创造历史最佳！({best_acc:.2f}% -> {current_accuracy:.2f}%) 提升了 {current_accuracy - best_acc:.2f}%")
            history["best_accuracy"] = current_accuracy
        else:
            print(f"📊 【准确率维稳】 当前 {current_accuracy:.2f}% (历史最佳为 {best_acc:.2f}%)")

        if avg_time_per_problem < best_time:
            print(f"⚡ 【速度突破】 推导与学习效率变快！(平均单题 {best_time:.2f} 秒 -> {avg_time_per_problem:.2f} 秒) 缩短了 {best_time - avg_time_per_problem:.2f} 秒")
            history["best_avg_time"] = avg_time_per_problem
        else:
            print(f"🐢 【速度维稳】 当前单题耗时 {avg_time_per_problem:.2f} 秒 (历史最佳为 {best_time:.2f} 秒)")

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

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    print("=============================================================\n")


if __name__ == "__main__":
    main()