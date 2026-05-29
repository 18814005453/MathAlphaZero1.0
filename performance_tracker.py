#!/usr/bin/env python3
"""
MathAlphaZero 性能追踪模块 v5.0

记录和展示训练过程中的能力增长：
- 每 N 个 epoch 自动跑 benchmark 评估
- 按难度分级统计 (easy/medium/hard)
- 生成训练曲线（文本图表）
- 对比训练前后的提升幅度
- 自动发现最佳模型
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PerformanceTracker:
    """追踪训练性能变化，展示能力增长"""

    def __init__(self, save_dir="data/performance"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history_path = os.path.join(save_dir, "performance_history.json")
        self.best_model_path = os.path.join(save_dir, "best_model_info.json")

        self.history = self._load(self.history_path, {
            "epochs": [],
            "benchmark": [],       # {"easy_acc": _, "medium_acc": _, "hard_acc": _, "weighted_score": _, "total_correct": _, "total_count": _}
            "training": [],        # {"loss_rule": _, "loss_loc": _, "loss_value": _, "success_rate": _}
            "rule_count": [],      # 规则数量随时间增长
            "best_score": 0,
            "best_epoch": 0,
        })

    def _load(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    # ===================== 记录 =====================

    def record_training(self, epoch: int, loss_rule: float, loss_loc: float,
                        loss_value: float, success_rate: float, rule_count: int):
        self.history["training"].append({
            "epoch": epoch,
            "loss_rule": loss_rule,
            "loss_loc": loss_loc,
            "loss_value": loss_value,
            "success_rate": success_rate,
            "rule_count": rule_count,
        })
        self.history["epochs"].append(epoch)
        self.history["rule_count"].append(rule_count)
        self._save()

    def record_benchmark(self, epoch: int, results: Dict[str, List[int]],
                         total_time: float):
        """
        results: {"easy": [correct, total], "medium": [correct, total], "hard": [correct, total]}
        """
        e_c, e_t = results.get("easy", [0, 0])
        m_c, m_t = results.get("medium", [0, 0])
        h_c, h_t = results.get("hard", [0, 0])

        total_correct = e_c + m_c + h_c
        total_count = e_t + m_t + h_t

        weights = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
        weighted_correct = e_c * 1.0 + m_c * 2.0 + h_c * 3.0
        weighted_total = e_t * 1.0 + m_t * 2.0 + h_t * 3.0
        weighted_score = (weighted_correct / weighted_total * 100) if weighted_total > 0 else 0

        entry = {
            "epoch": epoch,
            "easy_acc": (e_c / e_t * 100) if e_t > 0 else 0,
            "medium_acc": (m_c / m_t * 100) if m_t > 0 else 0,
            "hard_acc": (h_c / h_t * 100) if h_t > 0 else 0,
            "weighted_score": weighted_score,
            "total_correct": total_correct,
            "total_count": total_count,
            "total_time": total_time,
        }
        self.history["benchmark"].append(entry)

        # Track best model
        if weighted_score > self.history["best_score"]:
            self.history["best_score"] = weighted_score
            self.history["best_epoch"] = epoch
            print(f"\n  *** NEW BEST MODEL at epoch {epoch}: score={weighted_score:.2f} ***\n")

        self._save()

    def save_best_model_info(self, epoch: int, score: float, path: str):
        with open(self.best_model_path, 'w') as f:
            json.dump({"epoch": epoch, "score": score, "path": path}, f, indent=2)

    # ===================== 展示 =====================

    def show_training_curves(self) -> str:
        """生成 ASCII 训练曲线"""
        bench = self.history["benchmark"]
        train = self.history["training"]

        if len(bench) < 2:
            return "Not enough data for curves (need >= 2 evaluations)"

        lines = []
        width = 60

        # Benchmark score curve
        scores = [b["weighted_score"] for b in bench]
        epochs = [b["epoch"] for b in bench]
        lines.append("")
        lines.append("  BENCHMARK SCORE OVER TIME")
        lines.append("  " + "─" * width)

        min_s, max_s = min(scores), max(scores)
        s_range = max(max_s - min_s, 1)

        for i, (ep, sc) in enumerate(zip(epochs, scores)):
            bar_len = int((sc - min_s) / s_range * width)
            bar = "█" * bar_len
            marker = " ← NEW BEST" if sc == self.history["best_score"] else ""
            lines.append(f"  Epoch {ep:4d} │ {bar} {sc:.1f}{marker}")

        lines.append("  " + "─" * width)

        # Accuracy by difficulty
        if bench:
            lines.append("")
            lines.append("  ACCURACY BY DIFFICULTY LEVEL")
            lines.append("  " + "─" * width)
            header = f"  {'Epoch':<6} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Weighted':>10}"
            lines.append(header)
            lines.append("  " + "─" * width)
            for b in bench:
                lines.append(
                    f"  {b['epoch']:<6} {b['easy_acc']:>7.1f}% {b['medium_acc']:>7.1f}% "
                    f"{b['hard_acc']:>7.1f}% {b['weighted_score']:>9.1f}"
                )
            lines.append("  " + "─" * width)

        # Training loss curve
        if len(train) >= 2:
            lines.append("")
            lines.append("  TRAINING LOSS & SUCCESS RATE")
            lines.append("  " + "─" * width)
            header = f"  {'Epoch':<6} {'Loss(Rule)':>11} {'Loss(Loc)':>10} {'Loss(Val)':>10} {'Success':>8}"
            lines.append(header)
            lines.append("  " + "─" * width)

            # Show every N-th point
            step = max(1, len(train) // 15)
            for i in range(0, len(train), step):
                t = train[i]
                lines.append(
                    f"  {t['epoch']:<6} {t['loss_rule']:>10.4f} {t['loss_loc']:>9.4f} "
                    f"{t['loss_value']:>9.4f} {t['success_rate']:>7.1f}%"
                )
            lines.append("  " + "─" * width)

        return "\n".join(lines)

    def show_improvement_summary(self) -> str:
        """展示从训练开始到现在的提升幅度"""
        bench = self.history["benchmark"]
        train = self.history["training"]

        if len(bench) < 2:
            return "Need at least 2 evaluations to show improvement"

        first = bench[0]
        last = bench[-1]
        best_score = self.history["best_score"]

        lines = []
        lines.append("")
        lines.append("  ╔══════════════════════════════════════════════════════╗")
        lines.append("  ║        TRAINING IMPROVEMENT SUMMARY                 ║")
        lines.append("  ╠══════════════════════════════════════════════════════╣")

        # Score change
        delta = last["weighted_score"] - first["weighted_score"]
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        color = "✅" if delta > 0 else "⚠️" if delta < 0 else "➖"
        lines.append(f"  ║  Weighted Score:   {first['weighted_score']:6.1f}  →  {last['weighted_score']:6.1f}  "
                     f"({arrow} {delta:+.1f})  {color}   ║")
        lines.append(f"  ║  Best Score:       {best_score:6.1f}  (epoch {self.history['best_epoch']})                ║")

        # Per-difficulty improvement
        lines.append(f"  ╠══════════════════════════════════════════════════════╣")
        lines.append(f"  ║  {'Level':<10} {'Start':>8} {'Now':>8} {'Δ':>8} {'':<6} ║")
        for lvl in ["easy_acc", "medium_acc", "hard_acc"]:
            label = lvl.replace("_acc", "").title()
            d = last[lvl] - first[lvl]
            lines.append(f"  ║  {label:<10} {first[lvl]:>7.1f}% {last[lvl]:>7.1f}% {d:>+7.1f}%      ║")

        # Rule count growth
        if self.history["rule_count"]:
            first_rules = self.history["rule_count"][0]
            last_rules = self.history["rule_count"][-1]
            lines.append(f"  ╠══════════════════════════════════════════════════════╣")
            lines.append(f"  ║  Rule Library:     {first_rules:4d}  →  {last_rules:4d}  "
                         f"({last_rules - first_rules:+d} discovered)         ║")

        # Total problems solved
        first_correct = first["total_correct"]
        last_correct = last["total_correct"]
        lines.append(f"  ║  Problems Solved:  {first_correct:4d}  →  {last_correct:4d}                      ║")

        # Training progress
        if train:
            first_sr = train[0]["success_rate"] if train else 0
            last_sr = train[-1]["success_rate"] if train else 0
            lines.append(f"  ║  Train Success %:  {first_sr:5.1f}% → {last_sr:5.1f}%                      ║")

        lines.append("  ╚══════════════════════════════════════════════════════╝")
        lines.append("")
        return "\n".join(lines)

    def show_rule_contribution(self, net, preprocessor, device="cpu") -> str:
        """Analyze which rules contribute most to successful solves (from MCTS statistics)"""
        lines = []
        lines.append("")
        lines.append("  RULE USAGE HEATMAP (from recent evaluation)")
        lines.append("  " + "─" * 50)
        lines.append("  (Run a benchmark first to populate rule stats)")
        lines.append("  " + "─" * 50)
        return "\n".join(lines)

    def full_report(self) -> str:
        """生成完整报告"""
        parts = []
        parts.append(self.show_improvement_summary())
        parts.append(self.show_training_curves())
        return "\n".join(parts)


# ===================== 独立 Benchmark 评估函数 =====================

def run_benchmark_evaluation(net, preprocessor, device,
                             benchmark_path="benchmark_set.json",
                             max_problems=None, timeout_per_problem=15.0):
    """
    在完整 benchmark 上评估模型。
    返回 results dict 和 total_time
    """
    from core.state import IntegrationState, set_default_preprocessor
    from core.engine import MCTS

    set_default_preprocessor(preprocessor)

    if not os.path.exists(benchmark_path):
        print(f"Benchmark file not found: {benchmark_path}")
        return None, 0

    with open(benchmark_path, 'r') as f:
        dataset = json.load(f)

    if max_problems:
        dataset = dataset[:max_problems]

    x = sp.Symbol('x')
    results = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
    total_start = time.time()

    for item in dataset:
        lvl = item["difficulty"]
        results[lvl][1] += 1

        try:
            raw = sp.sympify(item["expression"])
            state = IntegrationState(expr=sp.Integral(raw, x))

            mcts = MCTS(
                network=net, preprocessor=preprocessor,
                num_simulations=100, max_depth=30,
                timeout=timeout_per_problem, device=device
            )
            traj = mcts.get_trajectory(state, temperature=0.0)

            if traj:
                last = traj[-1]
                ns, r, done, _ = mcts.env.step(last["state"], last["action"])
                if done and r > 0:
                    try:
                        target = sp.parse_expr(item["target_answer"])
                        diff = sp.simplify(sp.diff(ns.expr - target, x))
                        if diff == 0:
                            results[lvl][0] += 1
                    except Exception:
                        pass
        except Exception:
            pass

    total_time = time.time() - total_start
    return results, total_time


def evaluate_and_track(net, preprocessor, tracker: PerformanceTracker,
                       epoch: int, device="cpu", benchmark_path="benchmark_set.json"):
    """Evaluate model and record to tracker. Returns benchmark results dict."""
    print(f"\n  Running benchmark evaluation at epoch {epoch}...")
    results, total_time = run_benchmark_evaluation(
        net, preprocessor, device, benchmark_path=benchmark_path
    )

    if results is None:
        print("  Benchmark skipped (no dataset)")
        return None

    e_c, e_t = results["easy"]
    m_c, m_t = results["medium"]
    h_c, h_t = results["hard"]

    print(f"  Easy:   {e_c}/{e_t} ({e_c/e_t*100:.1f}%)" if e_t > 0 else "  Easy: 0/0")
    print(f"  Medium: {m_c}/{m_t} ({m_c/m_t*100:.1f}%)" if m_t > 0 else "  Medium: 0/0")
    print(f"  Hard:   {h_c}/{h_t} ({h_c/h_t*100:.1f}%)" if h_t > 0 else "  Hard: 0/0")

    tracker.record_benchmark(epoch, results, total_time)
    return results


# ===================== 独立运行 =====================

if __name__ == "__main__":
    import knowledge.rules
    from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules
    from core.network import MathNet
    from utils.preprocessor import MathPreprocessor

    build_action_space()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    preprocessor = MathPreprocessor(max_len=128)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()
    print(f"Rules: {num_rules}")

    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=128, nhead=4, num_layers=3, rule_num_layers=2,
        max_len=128, dropout=0.1, use_depth_embedding=True, max_depth=32
    ).to(device)
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))
    net.eval()

    # Run benchmark and show report
    tracker = PerformanceTracker()

    print("\n" + "=" * 70)
    print("  MathAlphaZero v5.0 — Performance Baseline")
    print("=" * 70)

    evaluate_and_track(net, preprocessor, tracker, epoch=0, device=str(device))

    # Simulate some training data for demo
    print("\n--- Simulating training progress ---")
    for ep in range(10, 110, 10):
        tracker.record_training(
            epoch=ep,
            loss_rule=1.0 - ep * 0.008,
            loss_loc=0.8 - ep * 0.006,
            loss_value=0.5 - ep * 0.004,
            success_rate=20 + ep * 0.6,
            rule_count=29 + ep // 20,
        )
        # Simulate improving benchmark
        fake_results = {
            "easy": [min(ep//2, 40), 40],
            "medium": [min(ep//3, 30), 30],
            "hard": [min(ep//5, 20), 30],
        }
        tracker.record_benchmark(ep, fake_results, ep * 2.0)

    print(tracker.full_report())

    print("\n  To use in training: from performance_tracker import PerformanceTracker, evaluate_and_track")
