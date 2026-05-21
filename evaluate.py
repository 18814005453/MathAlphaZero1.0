#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathAlphaZero 极速单进程评测流 (规避多进程管道死锁，支持成绩历史纵向对比)
位置：项目根目录下 evaluate.py
"""

import time
import json
import os
import sys
import torch
import sympy as sp
import warnings

# 强行闭嘴所有 PyTorch 的嵌套张量 Prototype 乱弹警告，还终端一片清爽
warnings.filterwarnings("ignore", category=UserWarning)

from knowledge.rules import MathRuleBase, RULE_NAMES
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from core.state import IntegrationState

def load_fixed_benchmark(file_path="benchmark_set.json"):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到数据集文件 '{file_path}'！")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_result(ai_result_expr, target_answer_str, x):
    try:
        if ai_result_expr is None: return False
        target_expr = sp.parse_expr(target_answer_str)
        diff_derivative = sp.simplify(sp.diff(ai_result_expr - target_expr, x))
        return diff_derivative == 0
    except Exception:
        return False

def run_benchmark_test(model_path="data/brain_final.pth"):
    dataset = load_fixed_benchmark()
    x = sp.Symbol('x')
    preprocessor = MathPreprocessor(max_len=128)
    rules = MathRuleBase()

    # 1. 核心提速：全局只加载一次模型，杜绝重复加载卡死
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=rules.num_actions,
        d_model=128, nhead=4, num_layers=3
    ).to(device)
    
    net.refresh_rule_cache(RULE_NAMES, tokenizer_fn=preprocessor._string_to_ids)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    report = {
        "easy": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0},
        "medium": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0},
        "hard": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0}
    }

    print("\n" + "=" * 70)
    print("🚀 MathAlphaZero 单进程极速流固定基准评测流水线已起跑")
    print("=" * 70 + "\n")

    for item in dataset:
        diff_level = item["difficulty"]
        expr_str = item["expression"]
        target = item["target_answer"]
        report[diff_level]["total"] += 1

        print(f"👉 [ID {item['id']:03d}] 难度: {diff_level.upper():<6} | 输入: ∫ {expr_str} dx")

        start_time = time.time()
        is_correct = False
        final_expr = None
        nodes_expanded = 0

        try:
            raw_expr = sp.sympify(expr_str)
            init_state = IntegrationState(expr=sp.Integral(raw_expr, x))
            
            # 使用 100 次探索快速评估基础算力
            mcts = MCTS(network=net, preprocessor=preprocessor, num_simulations=100)
            trajectory = mcts.get_trajectory(init_state, temperature=0.0)
            nodes_expanded = len(trajectory) * 100

            if trajectory:
                last_step = trajectory[-1]
                next_state_raw, reward, done, _ = mcts.env.step(last_step["state"], last_step["action"])
                if done and reward > 0:
                    final_expr = next_state_raw.expr
                    is_correct = verify_result(final_expr, target, x)
        except Exception as e:
            pass

        elapsed_time = time.time() - start_time
        report[diff_level]["total_time"] += elapsed_time

        if is_correct:
            report[diff_level]["correct"] += 1
            report[diff_level]["total_nodes"] += nodes_expanded
            print(f"   🟢 SUCCESS! 耗时: {elapsed_time:.3f}s | 答案: {final_expr}")
        else:
            print(f"   🔴 FAILED! 耗时: {elapsed_time:.3f}s")
        print("-" * 60)

    # 4. 全景得分大看盘
    print("\n🏆 MathAlphaZero 测试总报告 🏆")
    print("=" * 70)
    grand_total, grand_correct, grand_time = 0, 0, 0.0
    for lvl in ["easy", "medium", "hard"]:
        total = report[lvl]["total"]
        correct = report[lvl]["correct"]
        grand_total += total
        grand_correct += correct
        grand_time += report[lvl]["total_time"]
        acc = (correct / total * 100) if total > 0 else 0
        avg_t = (report[lvl]["total_time"] / total) if total > 0 else 0
        print(f"【{lvl.upper():<6}】 准确率: {acc:.1f}% ({correct}/{total}) | 平均耗时: {avg_t:.3f}s")

    overall_accuracy = (grand_correct / grand_total * 100) if grand_total > 0 else 0
    difficulty_weights = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
    weighted_correct = sum([difficulty_weights[l] * report[l]["correct"] for l in ["easy", "medium", "hard"]])
    weighted_total = sum([difficulty_weights[l] * report[l]["total"] for l in ["easy", "medium", "hard"]])
    final_score = (weighted_correct / weighted_total * 100.0) if weighted_total > 0 else 0.0

    print("\n" + "⭐" * 35)
    print(f"🎉 最终加权总分 (满分100) : {final_score:.2f} 分")
    print("⭐" * 35 + "\n")

    # 自动多轮趋势对比
    history_path = "data/benchmark_history.json"
    history_data = []
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, "r", encoding="utf-8") as f: history_data = json.load(f)
    if history_data:
        last_run = history_data[-1]
        print(f"⏳ 【相比上一次测试】：加权总分变动: {final_score - last_run['final_score']:.2f} 分")
    
    # 归档写入
    history_data.append({"final_score": final_score, "overall_accuracy": overall_accuracy, "total_time": grand_time})
    with open(history_path, "w", encoding="utf-8") as f: json.dump(history_data, f, indent=4)
    print("💾 成绩已自动封存入 data/benchmark_history.json\n")

if __name__ == "__main__":
    run_benchmark_test()
