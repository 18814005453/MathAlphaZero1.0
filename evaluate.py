#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathAlphaZero 工业级基准评测流水线 (带 20s 硬超时强杀机制)
位置：项目根目录下 evaluate.py
"""

import time
import json
import os
import sys
import multiprocessing
import torch
import sympy as sp
from sympy import parse_expr, diff, simplify

# =====================================================================
# 🔗 完美绑定你的核心架构组件
# =====================================================================
from knowledge.rules import MathRuleBase
from core.network import MathAlphaZeroNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from core.state import IntegrationState


def load_fixed_benchmark(file_path="benchmark_set.json"):
    """安全读取 200 道固定基准题库"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到数据集文件 '{file_path}'，请先运行生成脚本！")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_result(ai_result_expr, target_answer_str, x):
    """
    利用 SymPy 做连续域的微积分双向校验
    """
    try:
        if ai_result_expr is None:
            return False
        target_expr = parse_expr(target_answer_str)
        diff_derivative = simplify(diff(ai_result_expr - target_expr, x))
        if diff_derivative == 0:
            return True
        return False
    except Exception:
        return False


def worker_solve_task(expr_str, model_path, return_dict):
    """
    单道题目的子进程实际执行体
    """
    try:
        # 子进程内独立初始化网络和环境，防止多进程 CUDA 冲突（CPU无影响）
        x = sp.Symbol('x')
        preprocessor = MathPreprocessor(max_len=128)
        rules = MathRuleBase()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = MathAlphaZeroNet(
            vocab_size=preprocessor.vocab_size,
            num_actions=rules.num_actions,
            d_model=128,
            nhead=4,
            num_layers=3
        ).to(device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            net.load_state_dict(state_dict)
        net.eval()

        raw_expr = sp.sympify(expr_str)
        init_state = IntegrationState(expr=sp.Integral(raw_expr, x))

        mcts = MCTS(network=net, preprocessor=preprocessor, num_simulations=100)
        trajectory = mcts.get_trajectory(init_state, temperature=0.0)

        nodes_expanded = 0
        if hasattr(mcts, 'visit_count') and mcts.visit_count:
            nodes_expanded = len(mcts.visit_count)
        else:
            nodes_expanded = len(trajectory) * 100

        final_expr = None
        if trajectory:
            last_step = trajectory[-1]
            next_state_raw, reward, done, info = mcts.env.step(last_step["state"], last_step["action"])
            if done and reward > 0:
                final_expr = next_state_raw.expr

        # 将结果写回主进程共享字典
        return_dict["final_expr"] = final_expr
        return_dict["nodes_expanded"] = nodes_expanded
        return_dict["status"] = "success"

    except Exception as e:
        return_dict["status"] = f"error: {str(e)}"


def run_benchmark_test(model_path="data/brain.pth", timeout_limit=20.0):
    # 1. 加载 200 道固定数据集
    dataset = load_fixed_benchmark()
    x = sp.Symbol('x')

    # 2. 初始化多维度雷达统计看板
    report = {
        "easy": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0},
        "medium": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0},
        "hard": {"total": 0, "correct": 0, "total_time": 0.0, "total_nodes": 0}
    }

    print("\n" + "=" * 70)
    print("🚀 MathAlphaZero 工业级固定基准测试集 (200题完全体) 性能评测")
    print(f"⚠️  防卡死安全机制已启动：单题硬超时间隔 = {timeout_limit} 秒")
    print("=" * 70 + "\n")

    # 3. 使用多进程 Manager 管理跨进程数据通信
    manager = multiprocessing.Manager()

    for item in dataset:
        diff_level = item["difficulty"]
        expr_str = item["expression"]
        target = item["target_answer"]

        report[diff_level]["total"] += 1

        print(f"👉 [ID {item['id']:03d}] 难度: {diff_level.upper():<6} | 类型: {item['type']}")
        print(f"   输入: ∫ {expr_str} dx")

        # 为每一道题创建专属结果通信字典
        return_dict = manager.dict()
        return_dict["status"] = "running"
        return_dict["final_expr"] = None
        return_dict["nodes_expanded"] = 0

        # 启动沙盒多进程单独推导此题
        start_time = time.time()
        p = multiprocessing.Process(target=worker_solve_task, args=(expr_str, model_path, return_dict))
        p.start()

        # 主进程在此挂起，等待子进程，但最多只等 20 秒
        p.join(timeout=timeout_limit)

        elapsed_time = time.time() - start_time

        # --- ⏳ 超时强制拦截判定逻辑 ⏳ ---
        if p.is_alive():
            print(f"   🚨 TIME OUT! 耗时超过限制 ({timeout_limit}s)，强行终止该进程。")
            p.terminate()  # 强行杀死正在死循环的解题子进程
            p.join()  # 回收僵尸进程资源

            is_correct = False
            nodes_expanded = 0
            final_expr = None
            elapsed_time = timeout_limit  # 耗时按最大惩罚时间计算
        else:
            # 正常在 20 秒内跑完
            if return_dict["status"] == "success":
                final_expr = return_dict["final_expr"]
                nodes_expanded = return_dict["nodes_expanded"]
                # 触发严格的求导校验
                is_correct = verify_result(final_expr, target, x)
            else:
                is_correct = False
                nodes_expanded = 0
                final_expr = None

        # --- 📈 结果收录 ---
        if is_correct:
            report[diff_level]["correct"] += 1
            report[diff_level]["total_nodes"] += nodes_expanded
            print(f"   🟢 SUCCESS! 耗时: {elapsed_time:.3f}s | MCTS展开节点: {nodes_expanded}")
            print(f"   解出答案: {final_expr}")
        else:
            if return_dict["status"] != "success" and not return_dict["status"].startswith("running"):
                print(f"   🔴 FAILED! (语法报错/崩塌: {return_dict['status']})")
            elif return_dict["status"] == "running":
                pass  # 上面已经印过超时标记了
            else:
                print(f"   🔴 FAILED! (推导路径未通关 / 结果错误) 耗时: {elapsed_time:.3f}s")
        print("-" * 60)

        report[diff_level]["total_time"] += elapsed_time

    # 4. 打印最终全景得分看板
    print("\n" + "🏆 MathAlphaZero 固定数据集最终得分评测报告 🏆")
    print("=" * 70)

    grand_total = 0
    grand_correct = 0
    grand_time = 0.0

    for lvl in ["easy", "medium", "hard"]:
        total = report[lvl]["total"]
        correct = report[lvl]["correct"]
        grand_total += total
        grand_correct += correct
        grand_time += report[lvl]["total_time"]

        accuracy = (correct / total * 100) if total > 0 else 0
        avg_time = (report[lvl]["total_time"] / total) if total > 0 else 0
        avg_nodes = (report[lvl]["total_nodes"] / correct) if correct > 0 else 0

        print(f"难度级别 【{lvl.upper():<6}】 (共 {total} 道):")
        print(f"  🎯 准确率 (Accuracy) : {accuracy:.1f}% ({correct}/{total})")
        print(f"  ⚡ 平均耗时 (Latency)  : {avg_time:.3f} 秒")
        print(f"  🌲 均耗树节点 (Nodes)  : {avg_nodes:.1f} 个 (仅统计通关题型)")
        print("-" * 60)

    overall_accuracy = (grand_correct / grand_total * 100) if grand_total > 0 else 0
    print(f"🔥 全局总题库通关率 : {overall_accuracy:.2f}% ({grand_correct}/{grand_total})")
    print(f"⏱️ 200道题总运行耗时 : {grand_time:.2f} 秒")

    # ========== 新增：加权总分系统（满分100分） ==========
    difficulty_weights = {
        "easy": 1.0,
        "medium": 2.0,
        "hard": 3.0
    }
    weighted_correct = 0.0
    weighted_total = 0.0
    for lvl, weight in difficulty_weights.items():
        weighted_correct += weight * report[lvl]["correct"]
        weighted_total += weight * report[lvl]["total"]

    if weighted_total > 0:
        final_score = (weighted_correct / weighted_total) * 100.0
    else:
        final_score = 0.0

    print("\n" + "⭐" * 35)
    print(f"🏅 加权总分系统 (难度权重：Easy=1, Medium=2, Hard=3)")
    print(f"   加权得分 = {weighted_correct:.1f} / {weighted_total:.1f}")
    print(f"   🎉 最终总分 (满分100) : {final_score:.2f} 分")
    print("⭐" * 35 + "\n")
    # ===================================================

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 多进程在某些平台上（如 Windows/Mac）要求必须在 __main__ 中运行
    multiprocessing.freeze_support()
    run_benchmark_test()