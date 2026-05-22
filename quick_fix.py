# quick_fix.py - 快速修复版
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import random

# 必须先导入规则模块！
import knowledge.rules  # 这行很重要！

from core.state import IntegrationState
from core.env import IntegrationEnv
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules

print("=" * 60)
print("快速修复版训练")
print("=" * 60)

# 重建动作空间
build_action_space()
rule_names = get_all_rule_names()
num_actions = get_num_rules()

print(f"✅ 规则加载成功！动作空间大小: {num_actions}")
print(f"   规则示例: {rule_names[:5]}")

if num_actions == 0:
    print("❌ 错误：没有规则被加载，请检查 knowledge/rules.py")
    sys.exit(1)

# 初始化
preprocessor = MathPreprocessor(max_len=128)

# 创建网络
net = MathAlphaZeroNet(
    vocab_size=preprocessor.vocab_size,
    num_actions=num_actions,
    d_model=64,
    nhead=4,
    num_layers=2,
    learn_temperature=False
)
action_ids = list(range(num_actions))
net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids)
net.train()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# 简单经验池
buffer = []

# 测试题目
x = sp.Symbol('x')
test_problems = [
    sp.Integral(x ** 2, x),
    sp.Integral(x ** 3, x),
    sp.Integral(sp.sin(x), x),
    sp.Integral(sp.cos(x), x),
    sp.Integral(sp.exp(x), x),
]

print("\n开始训练...")
print("=" * 60)

for epoch in range(20):
    success_count = 0
    total_reward = 0

    for expr in test_problems:
        state = IntegrationState(expr)

        # 使用 MCTS
        mcts = MCTS(
            net, preprocessor,
            num_simulations=20,
            timeout=20,
            max_depth=10,
            num_parallel=1  # 禁用并行
        )

        # 获取轨迹
        try:
            trajectory = mcts.get_trajectory(state, temperature=0.5)

            if trajectory:
                # 检查是否成功
                last_step = trajectory[-1]
                env = IntegrationEnv(max_steps=10)
                next_state, reward, done, _ = env.step(last_step["state"], last_step["action"])

                if done and reward > 0:
                    success_count += 1
                    total_reward += reward

                    # 存储经验
                    for step in trajectory:
                        # 修复：正确获取 tensor 形状
                        state_tensor = preprocessor.state_to_tensor(step["state"].expr)  # shape: [1, max_len]
                        policy = torch.tensor(step["policy_target"], dtype=torch.float32)
                        value = torch.tensor([reward], dtype=torch.float32)
                        buffer.append((state_tensor, policy, value))
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 训练
    if len(buffer) >= 16:
        batch = random.sample(buffer, min(32, len(buffer)))
        states = torch.cat([b[0] for b in batch], dim=0)  # [batch, max_len]
        policies = torch.stack([b[1] for b in batch])
        values = torch.stack([b[2] for b in batch])

        # 创建掩码
        mask = torch.ones(states.shape[0], num_actions, dtype=torch.bool)

        policy_logits, pred_values = net(states, mask)

        # 损失
        log_probs = nn.LogSoftmax(dim=1)(policy_logits)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss = nn.MSELoss()(pred_values, values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
    else:
        loss_val = 0

    success_rate = success_count / len(test_problems) * 100
    print(f"Epoch {epoch + 1:2d} | 成功率: {success_rate:5.1f}% | 损失: {loss_val:.4f} | 经验池: {len(buffer)}")

    # 每5轮测试
    if (epoch + 1) % 5 == 0:
        print(f"  📈 平均奖励: {total_reward / len(test_problems):.2f}")

print("\n" + "=" * 60)
print("训练完成！")