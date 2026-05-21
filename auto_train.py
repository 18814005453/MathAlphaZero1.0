# auto_train.py
import os
import pickle
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp

# 将需要用到的内部模块统一在顶部导入
from core.rules import RULE_DICT, RULE_NAMES, MathRuleBase
from core.engine import MCTS
from core.network import MathAlphaZeroNet
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from core.state import IntegrationState


# ==================== 逆向生成法（保证可积性） ====================
def _random_coefficient():
    """生成随机系数（非零，约一半为整数，另一半为简单分数）"""
    if random.random() < 0.7:
        return random.randint(1, 5) * random.choice([-1, 1])
    else:
        num = random.randint(1, 3)
        den = random.randint(2, 4)
        return sp.Rational(num, den) * random.choice([-1, 1])

def _random_positive_integer(max_val=3):
    return random.randint(1, max_val)

def _generate_primitive_easy(x):
    """easy 难度原函数：单项式、基本初等函数"""
    choices = [
        x ** n for n in range(1, 4)                     # x, x^2, x^3
    ] + [
        sp.sin(k * x) for k in range(1, 3)              # sin(x), sin(2x)
    ] + [
        sp.cos(k * x) for k in range(1, 3)              # cos(x), cos(2x)
    ] + [
        sp.exp(k * x) for k in range(1, 3)              # e^x, e^{2x}
    ] + [
        sp.exp(-x)
    ]
    base = random.choice(choices)
    coeff = _random_coefficient()
    return coeff * base

def _generate_primitive_medium(x):
    """medium 难度原函数：乘积形式（需要分部积分）或简单复合"""
    prod_choices = [
        x * sp.sin(x), x * sp.cos(x), x * sp.exp(x),
        x**2 * sp.exp(x), x * sp.sin(2*x)
    ]
    comp_choices = [
        sp.sin(x**2), sp.exp(sp.sin(x)), sp.log(x+2), sp.atan(x)
    ]
    base = random.choice(prod_choices + comp_choices)
    coeff = _random_coefficient()
    return coeff * base

def _generate_primitive_hard(x):
    """hard 难度原函数：有理函数、嵌套、三角有理式等"""
    rational = [
        1/(x**2+1), x/(x**2+1), 1/((x+1)**2), sp.log(x**2+1)
    ]
    nested = [
        sp.exp(x**2), sp.sin(x**2), sp.cos(x**2), sp.exp(sp.sin(x))
    ]
    mixed = [
        x * sp.atan(x), x * sp.log(x+1)
    ]
    base = random.choice(rational + nested + mixed)
    coeff = _random_coefficient()
    poly_coeff = sp.Poly(random.randint(1,3)*x + random.randint(1,2), x)
    return coeff * poly_coeff * base

def generate_random_problem(difficulty: str = "easy") -> sp.Expr:
    """
    逆向生成法：先随机生成一个可积的原函数 F(x)，再返回其导数 f(x) = dF/dx
    保证 ∫ f(x) dx = F(x) + C 必定可积，且原函数已知。
    """
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
# ===============================================================


# ----------------------------- 经验池管理 -----------------------------
class ExperienceBuffer:
    """固定容量的经验回放池，存储 (state_tensor, policy_target, value_target)"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state_tensor, policy_target, value_target):
        """添加一条经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        state_cpu = state_tensor.cpu().clone()
        policy_cpu = torch.tensor(policy_target, dtype=torch.float32)
        value_cpu = torch.tensor([value_target], dtype=torch.float32)
        self.buffer[self.position] = (state_cpu, policy_cpu, value_cpu)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path: str):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'rb') as f:
                self.buffer = pickle.load(f)
                self.position = len(self.buffer) % self.capacity
                print(f"✅ 加载经验池，共 {len(self.buffer)} 条记录")
        else:
            print("ℹ️ 经验池为空或不存在，创建新经验池")


# ----------------------------- 训练主循环 -----------------------------
def main():
    print("====== MathAlphaZero 自我进化系统启动 ======")

    # 超参数配置
    MAX_SIMULATIONS = 80
    NUM_EPOCHS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SAVE_INTERVAL = 10

    # 初始化组件
    preprocessor = MathPreprocessor(max_len=128)
    rules = MathRuleBase()
    validator = MathValidator()

    # 创建神经网络
    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=rules.num_actions,
        d_model=128, nhead=4, num_layers=3
    )
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # 加载已有模型和经验池
    os.makedirs("data", exist_ok=True)
    if os.path.exists("data/brain.pth"):
        net.load_state_dict(torch.load("data/brain.pth"))
        print("✅ 加载已有大脑权重 (继承历史记忆)")
    memory = ExperienceBuffer(capacity=20000)
    memory.load("data/memory.pkl")

    # 训练统计
    solved_count = 0
    total_games = 0

    # 已解题目历史记录，防止重复学习同一道题
    solved_history = set()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- 世代 {epoch}/{NUM_EPOCHS} ---")

        # 动态课程学习：随着训练世代增加，逐渐解锁更高难度
        if epoch <= 50:
            DIFFICULTY = "easy"
        elif epoch <= 150:
            DIFFICULTY = "medium"
        else:
            DIFFICULTY = "hard"

        # 去重：最多尝试 50 次生成新题，避免重复
        for _ in range(50):
            expr = generate_random_problem(DIFFICULTY)
            if str(expr) not in solved_history:
                break
        else:
            print(f"⚠️ {DIFFICULTY} 难度的基础题型已基本学完！尝试强制生成新题。")
            expr = generate_random_problem("medium" if DIFFICULTY == "easy" else "hard")

        print(f"📝 探索新题目 [{DIFFICULTY}]: ∫ {expr} dx")
        total_games += 1

        # 包装初始状态
        x = sp.Symbol('x')
        init_state = IntegrationState(expr=sp.Integral(expr, x))

        # 实例化 MCTS
        mcts = MCTS(network=net, preprocessor=preprocessor, num_simulations=MAX_SIMULATIONS)

        # 获取完整训练轨迹
        trajectory = mcts.get_trajectory(init_state, temperature=1.0)

        # 解析轨迹结果
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
                path.append((final_expr, "Solved"))

        # 验证解是否正确
        if success and path:
            is_correct = validator.verify_integral(expr, final_expr)
            if not is_correct:
                print("⚠️ MCTS 宣称成功但验证失败，丢弃此轨迹")
                success = False

        # 去重：只存储新知识
        if success:
            expr_str = str(expr)
            if expr_str in solved_history:
                print(f"✅ 解题成功 (已掌握题型，跳过存储防过拟合)")
            else:
                solved_history.add(expr_str)
                solved_count += 1
                print(f"✅ 解题成功！路径长度: {len(path)} [✨ 解锁新知识]")
                for step_data in trajectory:
                    state_tensor = preprocessor.state_to_tensor(step_data["state"].expr)
                    memory.push(state_tensor, step_data["policy_target"], step_data["value_target"])
                print(f"📚 经验池容量扩展至: {len(memory)}")
        else:
            print("❌ 未找到有效解，需继续探索")

        # 神经网络训练
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

            print(f"🧠 训练更新: Loss = {total_loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")

        # 定期保存存档
        if epoch % SAVE_INTERVAL == 0:
            torch.save(net.state_dict(), "data/brain.pth")
            memory.save("data/memory.pkl")
            print(f"💾 存档已覆盖保存 (世代 {epoch})")

    # 训练结束最终保存
    torch.save(net.state_dict(), "data/brain_final.pth")
    memory.save("data/memory_final.pkl")
    print(f"\n🎉 阶段训练完成！新增掌握题型数: {solved_count}")


if __name__ == "__main__":
    main()