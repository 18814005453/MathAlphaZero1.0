# train_working.py - 基于修复版的工作训练脚本
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sympy as sp
import random
import json
import argparse
from collections import deque

# 必须导入规则
import knowledge.rules

from core.state import IntegrationState
from core.env import IntegrationEnv
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_model", type=str, default="data/brain_working.pth")
    args = parser.parse_args()

    print("=" * 60)
    print("MathAlphaZero 工作版训练")
    print("=" * 60)

    # 初始化
    build_action_space()
    rule_names = get_all_rule_names()
    num_actions = get_num_rules()
    print(f"✅ 动作空间: {num_actions} 个规则")

    preprocessor = MathPreprocessor(max_len=128)

    # 网络
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

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 经验池
    buffer = []

    # 题目集合
    x = sp.Symbol('x')
    problems = [
        sp.Integral(x ** 2, x),
        sp.Integral(x ** 3, x),
        sp.Integral(x ** 4, x),
        sp.Integral(sp.sin(x), x),
        sp.Integral(sp.cos(x), x),
        sp.Integral(sp.exp(x), x),
        sp.Integral(2 * x, x),
        sp.Integral(3 * x ** 2, x),
        sp.Integral(sp.sin(2 * x), x),
        sp.Integral(sp.cos(3 * x), x),
    ]

    history = {"epoch": [], "loss": [], "success_rate": []}

    for epoch in range(args.epochs):
        success_count = 0

        for expr in problems:
            state = IntegrationState(expr)

            mcts = MCTS(
                net, preprocessor,
                num_simulations=args.simulations,
                timeout=30,
                max_depth=15,
                num_parallel=1
            )

            trajectory = mcts.get_trajectory(state, temperature=0.5)

            if trajectory:
                last_step = trajectory[-1]
                env = IntegrationEnv(max_steps=10)
                next_state, reward, done, _ = env.step(last_step["state"], last_step["action"])

                if done and reward > 0:
                    success_count += 1
                    for step in trajectory:
                        state_tensor = preprocessor.state_to_tensor(step["state"].expr)
                        policy = torch.tensor(step["policy_target"], dtype=torch.float32)
                        value = torch.tensor([reward], dtype=torch.float32)
                        buffer.append((state_tensor, policy, value))

        # 训练
        if len(buffer) >= args.batch_size:
            batch = random.sample(buffer, min(args.batch_size, len(buffer)))
            states = torch.cat([b[0] for b in batch], dim=0)
            policies = torch.stack([b[1] for b in batch])
            values = torch.stack([b[2] for b in batch])
            mask = torch.ones(states.shape[0], num_actions, dtype=torch.bool)

            policy_logits, pred_values = net(states, mask)

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

        success_rate = success_count / len(problems) * 100
        history["epoch"].append(epoch + 1)
        history["loss"].append(loss_val)
        history["success_rate"].append(success_rate)

        print(f"Epoch {epoch + 1:3d} | 成功率: {success_rate:5.1f}% | 损失: {loss_val:.4f} | 经验池: {len(buffer)}")

        # 保存模型
        if (epoch + 1) % 20 == 0:
            torch.save(net.state_dict(), args.save_model)
            with open("data/history_working.json", "w") as f:
                json.dump(history, f, indent=4)

    torch.save(net.state_dict(), args.save_model)
    print(f"\n✅ 训练完成！模型保存至: {args.save_model}")


if __name__ == "__main__":
    main()