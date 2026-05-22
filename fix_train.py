# fix_train.py - 修复版训练脚本
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
from collections import deque
import random

# 导入模块
from core.state import IntegrationState
from core.env import IntegrationEnv
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules


# 简化版经验池（不使用PER，先保证能跑）
class SimpleBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, state_tensor, policy, value):
        self.buffer.append((state_tensor, policy, value))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def generate_simple_problem():
    """生成简单积分题目"""
    x = sp.Symbol('x')
    problems = [
        sp.Integral(x ** 2, x),
        sp.Integral(x ** 3, x),
        sp.Integral(sp.sin(x), x),
        sp.Integral(sp.cos(x), x),
        sp.Integral(sp.exp(x), x),
        sp.Integral(2 * x, x),
        sp.Integral(3 * x ** 2, x),
        sp.Integral(sp.sin(2 * x), x),
        sp.Integral(sp.cos(3 * x), x),
    ]
    return random.choice(problems)


def main():
    print("=" * 60)
    print("修复版训练 - 测试模型是否能够学习")
    print("=" * 60)

    # 初始化
    preprocessor = MathPreprocessor(max_len=128)
    build_action_space()
    rule_names = get_all_rule_names()
    num_actions = get_num_rules()

    print(f"动作空间: {num_actions}")
    print(f"规则示例: {rule_names[:5]}")

    # 创建网络
    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=num_actions,
        d_model=64,  # 减小模型尺寸加快训练
        nhead=4,
        num_layers=2,
        learn_temperature=False  # 固定温度
    )
    action_ids = list(range(num_actions))
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    buffer = SimpleBuffer(capacity=5000)

    # 测试几个简单题目
    test_problems = [
        sp.Integral(x ** 2, x),
        sp.Integral(sp.sin(x), x),
        sp.Integral(sp.exp(x), x),
    ]

    for epoch in range(30):
        epoch_loss = 0
        success_count = 0

        # 生成题目
        for _ in range(20):
            expr = generate_simple_problem()
            state = IntegrationState(expr)

            # MCTS搜索
            mcts = MCTS(
                net, preprocessor,
                num_simulations=30,  # 减少模拟次数
                timeout=30,
                max_depth=10,
                num_parallel=1  # 禁用并行
            )

            trajectory = mcts.get_trajectory(state, temperature=0.5)

            if trajectory:
                # 检查是否成功
                last_step = trajectory[-1]
                env = IntegrationEnv(max_steps=10)
                next_state, reward, done, _ = env.step(last_step["state"], last_step["action"])

                if done and reward > 0:
                    success_count += 1
                    # 存储经验
                    for step in trajectory:
                        state_tensor = preprocessor.state_to_tensor(step["state"].expr)
                        policy = torch.tensor(step["policy_target"], dtype=torch.float32)
                        value = torch.tensor([reward], dtype=torch.float32)
                        buffer.push(state_tensor.squeeze(0), policy, value)

        # 训练
        if len(buffer) >= 32:
            total_loss = 0
            for _ in range(10):
                batch = buffer.sample(32)
                states = torch.stack([b[0] for b in batch])
                policies = torch.stack([b[1] for b in batch])
                values = torch.stack([b[2] for b in batch])

                # 创建掩码（全合法）
                mask = torch.ones(states.shape[0], num_actions, dtype=torch.bool)

                policy_logits, pred_values = net(states, mask)

                # 损失计算
                log_probs = nn.LogSoftmax(dim=1)(policy_logits)
                policy_loss = -(policies * log_probs).sum(dim=1).mean()
                value_loss = nn.MSELoss()(pred_values, values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / 10
            epoch_loss = avg_loss

        success_rate = success_count / 20 * 100
        print(f"Epoch {epoch + 1:2d} | 成功率: {success_rate:5.1f}% | 损失: {epoch_loss:.4f} | 经验池: {len(buffer)}")

        # 每5轮测试一次
        if (epoch + 1) % 5 == 0:
            print("\n  📊 测试模型...")
            test_net(net, preprocessor, test_problems)
            print()


def test_net(net, preprocessor, problems):
    """测试网络是否能正确选择动作"""
    x = sp.Symbol('x')
    for expr in problems:
        state = IntegrationState(expr)
        state_tensor = preprocessor.state_to_tensor(expr.function).unsqueeze(0)
        mask = torch.ones(1, get_num_rules(), dtype=torch.bool)

        with torch.no_grad():
            policy, value = net(state_tensor, mask)
            probs = torch.softmax(policy, dim=1)
            top_prob, top_idx = probs.max(dim=1)

        print(f"    ∫ {expr.function} dx -> 动作: {top_prob[0]:.3f} | 价值: {value[0, 0]:.3f}")


if __name__ == "__main__":
    from sympy import Symbol

    x = Symbol('x')
    main()