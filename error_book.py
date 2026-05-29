#!/usr/bin/env python3
"""
错题本模块 — MathAlphaZero v7.0 核心组件

功能:
1. 训练中自动收集失败题目
2. 按失败原因分类 (规则不足 / 搜索不够 / 网络误判)
3. 错题优先级排序 (频率高+持续失败的优先)
4. 遗忘曲线: 连续做对 N 次→出本
5. 周期性错题重练 (30-50%的训练时间做错题)
6. 错题统计分析

设计理念:
- 人类学习: 做错的题反复练 → 高中状元
- AI学习: 成功样本满了经验池, 失败样本被丢弃 → 永远不进步
- 错题本: 把失败也变成训练信号
"""

import os
import json
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ErrorEntry:
    """单道错题的记录"""
    expr_str: str           # 积分表达式 (sympy 可解析)
    expr_hash: str          # 表达式 hash (用于去重)
    target_answer: str      # 标准答案
    difficulty: str         # easy/medium/hard
    fail_reason: str        # "no_rule" / "search_timeout" / "network_misjudge" / "verify_failed"
    fail_count: int = 1     # 累计失败次数
    total_attempts: int = 1 # 总尝试次数
    first_seen_epoch: int = 0
    last_seen_epoch: int = 0
    last_result: str = "fail"  # "fail" / "pass"
    streak_pass: int = 0    # 连续做对次数
    streak_fail: int = 1    # 连续失败次数
    complexity: int = 0     # AST 复杂度
    rules_blocked: List[str] = field(default_factory=list)  # 哪条规则匹配但执行失败


class ErrorBook:
    """
    错题本管理器
    """

    def __init__(self, save_path="data/error_book.json", max_errors=500):
        self.save_path = save_path
        self.max_errors = max_errors
        self.errors: Dict[str, ErrorEntry] = {}
        self._load()

    # ==================== 序列化 ====================

    def _load(self):
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            for item in data:
                e = ErrorEntry(
                    expr_str=item["expr_str"],
                    expr_hash=item["expr_hash"],
                    target_answer=item.get("target_answer", ""),
                    difficulty=item.get("difficulty", "easy"),
                    fail_reason=item.get("fail_reason", "unknown"),
                    fail_count=item.get("fail_count", 1),
                    total_attempts=item.get("total_attempts", 1),
                    first_seen_epoch=item.get("first_seen_epoch", 0),
                    last_seen_epoch=item.get("last_seen_epoch", 0),
                    last_result=item.get("last_result", "fail"),
                    streak_pass=item.get("streak_pass", 0),
                    streak_fail=item.get("streak_fail", 1),
                    complexity=item.get("complexity", 0),
                    rules_blocked=item.get("rules_blocked", []),
                )
                self.errors[e.expr_hash] = e
        except Exception:
            pass

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        data = []
        for e in sorted(self.errors.values(),
                        key=lambda x: (x.fail_count, x.streak_fail),
                        reverse=True):
            data.append({
                "expr_str": e.expr_str,
                "expr_hash": e.expr_hash,
                "target_answer": e.target_answer,
                "difficulty": e.difficulty,
                "fail_reason": e.fail_reason,
                "fail_count": e.fail_count,
                "total_attempts": e.total_attempts,
                "first_seen_epoch": e.first_seen_epoch,
                "last_seen_epoch": e.last_seen_epoch,
                "last_result": e.last_result,
                "streak_pass": e.streak_pass,
                "streak_fail": e.streak_fail,
                "complexity": e.complexity,
                "rules_blocked": e.rules_blocked,
            })
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ==================== CRUD ====================

    @staticmethod
    def _hash(expr_str: str) -> str:
        return hashlib.md5(expr_str.encode()).hexdigest()[:16]

    def record_fail(self, expr_str: str, target_answer: str = "",
                    difficulty: str = "easy", reason: str = "unknown",
                    complexity: int = 0, rules_blocked: List[str] = None,
                    epoch: int = 0):
        h = self._hash(expr_str)

        if h in self.errors:
            e = self.errors[h]
            e.fail_count += 1
            e.total_attempts += 1
            e.streak_fail += 1
            e.streak_pass = 0
            e.last_result = "fail"
            e.last_seen_epoch = epoch
            e.fail_reason = reason  # 更新最新失败原因
            if rules_blocked:
                e.rules_blocked = rules_blocked
        else:
            # 新错题
            if len(self.errors) >= self.max_errors:
                self._evict_one()

            e = ErrorEntry(
                expr_str=expr_str,
                expr_hash=h,
                target_answer=target_answer,
                difficulty=difficulty,
                fail_reason=reason,
                first_seen_epoch=epoch,
                last_seen_epoch=epoch,
                complexity=complexity,
                rules_blocked=rules_blocked or [],
            )
            self.errors[h] = e

        self._save()
        return e

    def record_pass(self, expr_str: str, epoch: int = 0):
        h = self._hash(expr_str)
        if h not in self.errors:
            return

        e = self.errors[h]
        e.total_attempts += 1
        e.streak_pass += 1
        e.streak_fail = 0
        e.last_result = "pass"
        e.last_seen_epoch = epoch

        # 连续做对 3 次 → 移出错题本
        if e.streak_pass >= 3:
            del self.errors[h]

        self._save()

    def _evict_one(self):
        """移除最不重要的错题 (连续失败最少的)"""
        if not self.errors:
            return
        worst = min(self.errors.values(),
                   key=lambda e: (e.streak_fail, e.fail_count, -e.total_attempts))
        del self.errors[worst.expr_hash]

    # ==================== 查询 ====================

    def is_error(self, expr_str: str) -> bool:
        return self._hash(expr_str) in self.errors

    def get_hardest(self, n: int = 10) -> List[ErrorEntry]:
        """最需要突破的错题 (连续失败多 + 总失败多)"""
        return sorted(self.errors.values(),
                     key=lambda e: (e.streak_fail, e.fail_count),
                     reverse=True)[:n]

    def sample_review_set(self, n: int = 20, bias_recent: bool = True) -> List[ErrorEntry]:
        """
        抽取复习题集。
        bias_recent=True: 60%新错题 + 40%顽固错题
        """
        if not self.errors:
            return []

        errors = list(self.errors.values())

        if bias_recent and len(errors) > 5:
            # 新错题 (最近 5 个 epoch 内的)
            recent = [e for e in errors if e.streak_fail >= 2]
            hardened = [e for e in errors if e.streak_fail >= 4]

            import random
            n_recent = min(len(recent), int(n * 0.6))
            n_hard = min(len(hardened), n - n_recent)
            n_rest = n - n_recent - n_hard

            sample = []
            if recent:
                sample += random.sample(recent, n_recent)
            if hardened:
                sample += random.sample(hardened, n_hard)
            if n_rest > 0:
                rest = [e for e in errors if e not in sample]
                if rest:
                    sample += random.sample(rest, min(n_rest, len(rest)))
            return sample

        import random
        return random.sample(errors, min(n, len(errors)))

    # ==================== 分析 ====================

    def stats(self) -> Dict:
        """错题本统计分析"""
        if not self.errors:
            return {"total": 0, "message": "No errors recorded yet"}

        errors = list(self.errors.values())

        by_reason = defaultdict(int)
        by_difficulty = defaultdict(int)
        by_rule = defaultdict(int)

        for e in errors:
            by_reason[e.fail_reason] += 1
            by_difficulty[e.difficulty] += 1
            for rule in e.rules_blocked:
                by_rule[rule] += 1

        return {
            "total": len(errors),
            "total_fail_count": sum(e.fail_count for e in errors),
            "by_reason": dict(by_reason),
            "by_difficulty": dict(by_difficulty),
            "most_blocked_rules": sorted(by_rule.items(), key=lambda x: -x[1])[:5],
            "hardest": [
                {"expr": e.expr_str, "streak_fail": e.streak_fail, "fail_count": e.fail_count}
                for e in self.get_hardest(5)
            ],
            "avg_streak_fail": sum(e.streak_fail for e in errors) / len(errors),
            "mastered_count": sum(1 for e in errors if e.streak_pass >= 2),
        }

    def print_report(self):
        """打印错题分析报告"""
        s = self.stats()
        print(f"\n  {'='*50}")
        print(f"  ERROR BOOK REPORT")
        print(f"  {'='*50}")
        print(f"  Total errors:     {s['total']}")
        print(f"  Total failures:   {s['total_fail_count']}")
        print(f"  Avg streak fail:  {s['avg_streak_fail']:.1f}")
        print(f"  Near master:      {s['mastered_count']}")
        print()
        print(f"  By reason:")
        for reason, count in s.get('by_reason', {}).items():
            bar = '█' * int(count / max(1, s['total']) * 30)
            print(f"    {reason:<25s} {bar} {count}")
        print()
        print(f"  By difficulty:")
        for diff, count in s.get('by_difficulty', {}).items():
            print(f"    {diff:<10s} {count}")
        print()
        if s.get('most_blocked_rules'):
            print(f"  Rules most often blocking progress:")
            for rule, count in s['most_blocked_rules']:
                print(f"    {rule:<40s} {count} failures")
        print()
        if s.get('hardest'):
            print(f"  Hardest problems (need most work):")
            for e in s['hardest']:
                print(f"    {e['expr'][:50]:<50s} failed {e['streak_fail']}x in a row")
        print(f"  {'='*50}\n")


if __name__ == "__main__":
    # Demo
    eb = ErrorBook(save_path="/tmp/test_error_book.json")

    # Simulate training
    print("Simulating training with error book...\n")

    problems = [
        ("x**2", "x**3/3", "easy"),
        ("sin(x)**2", "x/2 - sin(2*x)/4", "medium"),
        ("1/(1+x**3)", "log(x+1)/3 - log(x**2-x+1)/6 + sqrt(3)*atan((2*x-1)/sqrt(3))/3", "hard"),
        ("x*sin(x)", "sin(x) - x*cos(x)", "medium"),
        ("exp(x)*sin(x)", "(sin(x)-cos(x))*exp(x)/2", "medium"),
    ]

    for epoch in range(10):
        print(f"--- Epoch {epoch+1} ---")

        # 错题重练
        review = eb.sample_review_set(n=3)
        if review:
            print(f"  Reviewing {len(review)} errors: {[e.expr_str for e in review]}")
            for e in review:
                if "sin(x)**2" in e.expr_str and epoch >= 3:
                    # 训练后终于做对了
                    eb.record_pass(e.expr_str, epoch=epoch)
                    print(f"    PASSED {e.expr_str}! (mastered after {e.streak_fail} fails)")
                elif "x*sin(x)" in e.expr_str and epoch >= 1:
                    eb.record_pass(e.expr_str, epoch=epoch)
                    print(f"    PASSED {e.expr_str}!")

        # 新题失败记录
        for expr, target, diff in problems:
            if random.random() < 0.3:  # 30% 失败率
                reason = random.choice(["network_misjudge", "search_timeout", "no_rule"])
                eb.record_fail(expr, target, diff, reason, epoch=epoch)
                print(f"  FAILED {expr} ({reason})")

    eb.print_report()

    def random():
        pass
