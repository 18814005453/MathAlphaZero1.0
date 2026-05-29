#!/usr/bin/env python3
"""MathAlphaZero 极速评测流 — 重构版"""

import time
import json
import os
import sys
import torch
import sympy as sp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from core.network import MathNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from core.state import IntegrationState, set_default_preprocessor
from knowledge.rule_registry import get_all_rule_names, get_num_rules, build_action_space
import knowledge.rules  # 触发规则注册


def load_benchmark(file_path="benchmark_set.json"):
    if not os.path.exists(file_path):
        print(f"ERROR: cannot find '{file_path}'")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_result(ai_result_expr, target_answer_str, x):
    try:
        if ai_result_expr is None:
            return False
        target_expr = sp.parse_expr(target_answer_str)
        diff = sp.simplify(sp.diff(ai_result_expr - target_expr, x))
        return diff == 0
    except Exception:
        return False


def run_benchmark_test(model_path="data/brain.pth"):
    dataset = load_benchmark()
    x = sp.Symbol('x')

    build_action_space()
    preprocessor = MathPreprocessor(max_len=128)
    set_default_preprocessor(preprocessor)
    rule_names = get_all_rule_names()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=128, nhead=4, num_layers=3,
        rule_num_layers=2, max_len=128, dropout=0.1,
        use_depth_embedding=True, max_depth=32
    ).to(device)

    action_ids = list(range(get_num_rules()))
    net.refresh_rule_cache(rule_names, preprocessor.tokenize_list, action_ids=action_ids)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
    net.eval()

    report = {
        "easy": {"total": 0, "correct": 0, "total_time": 0.0},
        "medium": {"total": 0, "correct": 0, "total_time": 0.0},
        "hard": {"total": 0, "correct": 0, "total_time": 0.0},
    }

    print("\n" + "=" * 70)
    print("MathAlphaZero Benchmark Evaluation")
    print("=" * 70 + "\n")

    for item in dataset:
        diff_level = item["difficulty"]
        expr_str = item["expression"]
        target = item["target_answer"]
        report[diff_level]["total"] += 1

        print(f"  [ID {item['id']:03d}] {diff_level.upper():<6} | integrand: {expr_str}")

        start_time = time.time()
        is_correct = False

        try:
            raw_expr = sp.sympify(expr_str)
            init_state = IntegrationState(expr=sp.Integral(raw_expr, x))

            mcts = MCTS(network=net, preprocessor=preprocessor,
                        num_simulations=100, device=device)
            trajectory = mcts.get_trajectory(init_state, temperature=0.0)

            if trajectory:
                last_step = trajectory[-1]
                next_state, reward, done, _ = mcts.env.step(
                    last_step["state"], last_step["action"]
                )
                if done and reward > 0:
                    is_correct = verify_result(next_state.expr, target, x)
        except Exception:
            pass

        elapsed = time.time() - start_time
        report[diff_level]["total_time"] += elapsed

        if is_correct:
            report[diff_level]["correct"] += 1
            print(f"     SUCCESS  {elapsed:.3f}s")
        else:
            print(f"     FAILED  {elapsed:.3f}s")
        print("-" * 60)

    # 总报告
    print("\n" + "=" * 70)
    print("Evaluation Report")
    print("=" * 70)
    grand_total = grand_correct = grand_time = 0
    weights = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
    weighted_correct = weighted_total = 0.0

    for lvl in ["easy", "medium", "hard"]:
        t = report[lvl]["total"]
        c = report[lvl]["correct"]
        grand_total += t
        grand_correct += c
        grand_time += report[lvl]["total_time"]
        weighted_correct += weights[lvl] * c
        weighted_total += weights[lvl] * t
        acc = (c / t * 100) if t > 0 else 0
        avg_t = (report[lvl]["total_time"] / t) if t > 0 else 0
        print(f"  [{lvl.upper():<6}] accuracy: {acc:.1f}% ({c}/{t})  avg: {avg_t:.3f}s")

    overall = (grand_correct / grand_total * 100) if grand_total > 0 else 0
    final_score = (weighted_correct / weighted_total * 100.0) if weighted_total > 0 else 0.0

    print(f"\n  Overall accuracy: {overall:.2f}%")
    print(f"  Weighted score:   {final_score:.2f}/100")
    print("=" * 70 + "\n")

    # 历史对比
    history_path = "data/benchmark_history.json"
    history_data = []
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, "r") as f:
            history_data = json.load(f)
    if history_data:
        last = history_data[-1]
        delta = final_score - last["final_score"]
        print(f"  vs previous: {delta:+.2f} pts")

    history_data.append({
        "final_score": final_score,
        "overall_accuracy": overall,
        "total_time": grand_time,
    })
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=4)
    print("  Results saved to data/benchmark_history.json\n")


if __name__ == "__main__":
    run_benchmark_test()
