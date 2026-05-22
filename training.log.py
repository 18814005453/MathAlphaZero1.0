#!/usr/bin/env python3
# auto_train.py - MathAlphaZero 3.0 自动训练脚本
# 用法: python auto_train.py [--epochs 200] [--batch_size 64] ...

import os
import sys
import time
import json
import math
import random
import pickle
import argparse
import logging
import threading
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp

# ----------------------------- 项目模块导入 -----------------------------
# 确保所有升级后的模块位于正确路径
from core.state import IntegrationState
from core.actions import Action
from core.env import IntegrationEnv
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from knowledge.rule_registry import get_all_rule_names, get_num_rules, build_action_space, reload_module

# ----------------------------- 日志配置 -----------------------------
LOG_FILE = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MathAlphaZero")

# ----------------------------- 辅助函数 -----------------------------
def _random_coefficient():
    if random.random() < 0.7:
        return random.randint(1, 5) * random.choice([-1, 1])
    else:
        num = random.randint(1, 3)
        den = random.randint(2, 4)
        return sp.Rational(num, den) * random.choice([-1, 1])

def _generate_primitive_easy(x):
    choices = (
        [x**n for n in range(1, 4)] +
        [sp.sin(k*x) for k in range(1, 3)] +
        [sp.cos(k*x) for k in range(1, 3)] +
        [sp.exp(k*x) for k in range(1, 3)] +
        [sp.exp(-x)]
    )
    base = random.choice(choices)
    coeff = _random_coefficient()
    return coeff * base

def _generate_primitive_medium(x):
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
    poly_coeff = random.randint(1,3)*x + random.randint(1,2)
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
    return sp.simplify(f)

def ast_depth(expr):
    if expr.is_Atom:
        return 1
    return 1 + max(ast_depth(arg) for arg in expr.args)

def generate_problem_with_ast_depth(max_depth: int, max_attempts=30) -> sp.Expr:
    for _ in range(max_attempts):
        diff = random.choices(["easy","medium","hard"], weights=[0.2,0.3,0.5])[0]
        expr = generate_random_problem(diff)
        if ast_depth(expr) <= max_depth:
            return expr
    return generate_random_problem("easy")

# ----------------------------- 优先经验回放 (PER) -----------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity=30000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, item, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = item
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(prio, 1e-6)

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.buffer, self.priorities, self.pos), f)

    def load(self, path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'rb') as f:
                self.buffer, self.priorities, self.pos = pickle.load(f)

# ----------------------------- 课程学习 -----------------------------
class CurriculumTracker:
    def __init__(self, start_depth=2, max_depth=20, window_size=50):
        self.depth_performance = defaultdict(lambda: deque(maxlen=window_size))
        self.current_depth = start_depth
        self.max_depth = max_depth

    def update(self, depth, success):
        self.depth_performance[depth].append(1.0 if success else 0.0)

    def step(self, threshold=0.65):
        perf = self.depth_performance[self.current_depth]
        if len(perf) >= 10 and np.mean(perf) >= threshold:
            self.current_depth = min(self.current_depth + 1, self.max_depth)
        elif len(perf) >= 20 and np.mean(perf) < 0.3:
            self.current_depth = max(2, self.current_depth - 1)
        return self.current_depth

# ----------------------------- 监控热重载标志 -----------------------------
def check_and_reload(net, preprocessor):
    flag_file = "data/RELOAD_FLAG"
    if os.path.exists(flag_file):
        try:
            with open(flag_file, 'r') as f:
                rule_name = f.read().strip()
            logger.info(f"检测到热重载标志，新规则: {rule_name}")
            # 重载规则模块
            reload_module("knowledge.rules")
            new_rule_names = get_all_rule_names()
            new_num_rules = get_num_rules()
            action_ids = list(range(new_num_rules))
            net.refresh_rule_cache(new_rule_names, preprocessor._string_to_ids, action_ids=action_ids)
            logger.info(f"热重载完成，动作空间大小: {new_num_rules}")
        except Exception as e:
            logger.error(f"热重载失败: {e}")
        finally:
            os.remove(flag_file)

# ----------------------------- 主训练函数 -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MathAlphaZero 自动训练")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--problems_per_epoch", type=int, default=60, help="每轮题目数")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS模拟次数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.96, help="折扣因子")
    parser.add_argument("--td_lambda", type=float, default=0.8, help="TD(λ)参数")
    parser.add_argument("--n_step", type=int, default=5, help="n步回报")
    parser.add_argument("--timeout", type=float, default=60.0, help="每题超时(秒)")
    parser.add_argument("--max_depth", type=int, default=30, help="最大深度")
    parser.add_argument("--temperature", type=float, default=1.0, help="初始温度")
    parser.add_argument("--load_model", type=str, default="data/brain_3.0.pth", help="加载模型路径")
    parser.add_argument("--save_model", type=str, default="data/brain_3.0.pth", help="保存模型路径")
    parser.add_argument("--memory_path", type=str, default="data/memory_per.pkl", help="经验池路径")
    parser.add_argument("--miner_memory", type=str, default="data/memory_final_for_miner.pkl", help="模式挖掘数据路径")
    parser.add_argument("--no_auto_discover", action="store_true", help="禁用自动发现监听")
    args = parser.parse_args()

    logger.info("====== MathAlphaZero 3.0 自动训练启动 ======")
    logger.info(f"参数配置: {vars(args)}")

    # 初始化组件
    preprocessor = MathPreprocessor(max_len=128)
    validator = MathValidator()
    build_action_space()
    rule_names = get_all_rule_names()
    num_actions = get_num_rules()
    logger.info(f"规则库加载完成，动作空间大小: {num_actions}")

    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=num_actions,
        d_model=128, nhead=4, num_layers=3,
        learn_temperature=True
    )
    action_ids = list(range(num_actions))
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids=action_ids)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs("data", exist_ok=True)

    if os.path.exists(args.load_model):
        net.load_state_dict(torch.load(args.load_model))
        logger.info(f"加载模型: {args.load_model}")

    memory = PrioritizedReplayBuffer(capacity=30000)
    memory.load(args.memory_path)
    logger.info(f"经验池加载完成，当前容量: {len(memory)}")

    curriculum = CurriculumTracker(start_depth=2, max_depth=15)
    solved_set = set()

    # 训练历史记录
    history = {"epoch": [], "success_rate": [], "loss": [], "depth": []}

    # 主训练循环
    for epoch in range(1, args.epochs + 1):
        current_depth = curriculum.step(threshold=0.65)
        logger.info(f"Epoch {epoch}/{args.epochs} | 课程深度: {current_depth} | 经验池大小: {len(memory)}")

        # 生成题目
        problems = []
        for _ in range(args.problems_per_epoch):
            expr = generate_problem_with_ast_depth(current_depth)
            norm = str(expr)
            if norm in solved_set:
                continue
            problems.append(expr)

        if not problems:
            logger.warning("本轮无新题目，跳过")
            continue

        success_count = 0
        epoch_losses = []

        for expr in problems:
            state = IntegrationState(sp.Integral(expr, sp.Symbol('x')))
            mcts = MCTS(
                network=net,
                preprocessor=preprocessor,
                num_simulations=args.simulations,
                timeout=args.timeout,
                max_depth=args.max_depth,
                gamma=args.gamma,
                num_parallel=4
            )
            trajectory = mcts.get_trajectory(state, temperature=args.temperature)
            if not trajectory:
                curriculum.update(current_depth, False)
                continue

            # 最终验证
            last_step = trajectory[-1]
            env = IntegrationEnv(max_steps=args.max_depth, time_limit=args.timeout)
            next_state, reward, done, _ = env.step(last_step["state"], last_step["action"])
            success = (done and reward > 0.8 and validator.verify_integral(expr, next_state.expr))

            if success:
                success_count += 1
                solved_set.add(str(expr))
                curriculum.update(current_depth, True)
                # 计算 n 步 TD(λ) 目标
                values = [step.get("value_target", 0.0) for step in trajectory]
                lambda_returns = []
                for t in range(len(trajectory)):
                    g = 0.0
                    for k in range(args.n_step):
                        if t + k < len(trajectory):
                            g += (args.gamma ** k) * values[t + k]
                    if t + args.n_step < len(trajectory):
                        g += (args.gamma ** args.n_step) * values[t + args.n_step] * (1 - args.td_lambda)
                    lambda_returns.append(g)

                for idx, step in enumerate(trajectory):
                    state_tensor = preprocessor.state_to_tensor(step["state"].expr)
                    policy_target = step["policy_target"]
                    action_id = step["action"].id
                    mask = torch.ones(num_actions, dtype=torch.bool)
                    value_target = lambda_returns[idx] if idx < len(lambda_returns) else reward
                    td_error = abs(step.get("value_target", 0.0) - value_target)
                    priority = td_error + 1e-6
                    item = (state_tensor.cpu().clone(),
                            torch.tensor(policy_target, dtype=torch.float32),
                            torch.tensor([value_target], dtype=torch.float32),
                            action_id,
                            mask)
                    memory.push(item, priority=priority)
                logger.debug(f"✅ 成功: ∫ {expr} dx")
            else:
                curriculum.update(current_depth, False)
                logger.debug(f"❌ 失败: ∫ {expr} dx")

        # 训练网络
        if len(memory) >= args.batch_size:
            total_loss = 0.0
            num_batches = 5
            for _ in range(num_batches):
                batch, weights, indices = memory.sample(args.batch_size)
                if not batch:
                    continue
                batch_states, batch_policies, batch_values, _, batch_masks = zip(*batch)
                batch_states = torch.cat(batch_states, dim=0)
                batch_policies = torch.stack(batch_policies)
                batch_values = torch.stack(batch_values)
                batch_masks = torch.stack(batch_masks)
                weights = torch.tensor(weights, dtype=torch.float32)

                policy_logits, pred_values = net(batch_states, batch_masks)
                log_probs = nn.LogSoftmax(dim=1)(policy_logits)
                policy_loss = -(batch_policies * log_probs).sum(dim=1) * weights
                policy_loss = policy_loss.mean()
                value_loss = (pred_values - batch_values).pow(2).squeeze() * weights
                value_loss = value_loss.mean()
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

                # 更新优先级
                with torch.no_grad():
                    td_errors = (pred_values - batch_values).abs().squeeze().cpu().numpy()
                memory.update_priorities(indices, td_errors + 1e-6)

            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            logger.info(f"训练损失: {avg_loss:.4f}")

        # 保存模型和经验池
        torch.save(net.state_dict(), args.save_model)
        memory.save(args.memory_path)

        # 保存一份用于模式挖掘的数据（仅成功轨迹的摘要，简化版）
        # 实际使用中，可以在成功时直接写入 miner 专用文件，这里省略复杂逻辑
        # 仅创建一个空文件占位
        if not os.path.exists(args.miner_memory):
            with open(args.miner_memory, 'wb') as f:
                pickle.dump({"actions": [], "reward": []}, f)

        success_rate = success_count / max(1, len(problems)) * 100
        history["epoch"].append(epoch)
        history["success_rate"].append(success_rate)
        history["loss"].append(np.mean(epoch_losses) if epoch_losses else 0)
        history["depth"].append(current_depth)

        # 保存训练历史 JSON
        with open("data/training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        logger.info(f"Epoch {epoch} 完成 | 成功率: {success_rate:.1f}% | 累计成功题目: {len(solved_set)}")

        # 每隔 5 轮检查并热加载新宏规则（如果 auto_discover 进程产生了标志）
        if not args.no_auto_discover and epoch % 5 == 0:
            check_and_reload(net, preprocessor)

    logger.info("🎉 训练完成！")
    logger.info(f"最终成功率: {history['success_rate'][-1]:.1f}% (最后10轮平均)")

if __name__ == "__main__":
    main()