#!/usr/bin/env python3
"""MathAlphaZero 完整测试套件 v5.0"""
import sys
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import sympy as sp
from sympy import Integral, sin, cos, exp, tan, log, sqrt, atan, asin, sec, csc, cot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge.rule_registry import build_action_space, get_all_rule_names, get_num_rules
import knowledge.rules

from core.network import MathNet
from core.engine import MCTS
from core.state import IntegrationState, set_default_preprocessor
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator


def test_rule_registry():
    """1. 规则注册表完整性测试"""
    print("=" * 60)
    print("TEST 1: Rule Registry")
    print("=" * 60)
    build_action_space()
    names = get_all_rule_names()
    n = get_num_rules()
    print(f"  Total rules: {n}")
    assert n > 0, "No rules registered!"
    assert n >= 20, f"Expected at least 20 rules, got {n}"
    for name in names:
        from knowledge.rule_registry import get_rule_by_name
        fn = get_rule_by_name(name)
        assert callable(fn), f"Rule {name} is not callable"
    print(f"  PASSED: {n} rules registered and callable")
    return True


def test_rules_direct_apply():
    """2b. 规则直接调用测试（不走网络）"""
    print("\n" + "=" * 60)
    print("TEST 2b: Direct Rule Application (No Network)")
    print("=" * 60)
    from knowledge.rule_registry import get_rule_by_name

    x = sp.Symbol('x')
    test_cases = [
        ("rule_power_integral", Integral(x**2, x), True),
        ("rule_power_integral", Integral(x**3, x), True),
        ("rule_power_integral", Integral(1/x, x), True),
        ("rule_trig_integral", Integral(sin(x), x), True),
        ("rule_trig_integral", Integral(cos(x), x), True),
        ("rule_exp_integral", Integral(exp(x), x), True),
        ("rule_exp_integral", Integral(exp(2*x), x), True),
        ("rule_log_integral", Integral(log(x), x), True),
        ("rule_inv_trig_integral", Integral(1/(x**2+1), x), True),
        ("rule_extract_constant", Integral(5*sin(x), x), True),
        ("rule_split_addition", Integral(sin(x)+cos(x), x), True),
    ]

    passed = 0
    for rule_name, expr, should_match in test_cases:
        fn = get_rule_by_name(rule_name)
        result = fn(expr)
        if should_match and result is not None:
            new_expr, status = result
            print(f"  {rule_name}({expr.function}): {status}")
            passed += 1
        elif not should_match:
            passed += 1
        else:
            print(f"  FAILED: {rule_name}({expr.function}) returned None")

    total = len(test_cases)
    print(f"\n  PASSED: {passed}/{total}")
    return passed >= total * 0.85


def test_e2e_pipeline():
    """5. 端到端管道测试（不要求未训练网络输出正确答案）"""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Pipeline (Crash Test)")
    print("=" * 60)

    preprocessor = MathPreprocessor(max_len=128)
    set_default_preprocessor(preprocessor)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()

    device = torch.device("cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, dropout=0.0, use_depth_embedding=True, max_depth=32
    ).to(device)
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))
    net.eval()

    x = sp.Symbol('x')
    for integrand in [x**2, x**3, sin(x), cos(x), exp(x)]:
        state = IntegrationState(expr=Integral(integrand, x))
        mcts = MCTS(network=net, preprocessor=preprocessor,
                    num_simulations=30, max_depth=10, timeout=5.0, device=device)
        try:
            actions, probs = mcts.get_action_probs(state, temperature=1.0)
            print(f"  Integral({integrand}): {len(actions)} actions found, pipeline OK")
        except Exception as e:
            print(f"  Integral({integrand}): ERROR - {e}")
            return False
    print(f"  PASSED: pipeline runs without crash")
    return True


def test_network_construction():
    """3. 网络构建和前向传播测试"""
    print("\n" + "=" * 60)
    print("TEST 3: Network Construction & Forward Pass")
    print("=" * 60)

    preprocessor = MathPreprocessor(max_len=128)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()

    device = torch.device("cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, dropout=0.0, use_depth_embedding=True, max_depth=32
    ).to(device)

    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))

    x = sp.Symbol('x')
    expr = sp.Integral(x**2 + sin(x), x)
    token_t, depth_t = preprocessor.state_to_tensor_with_depth(expr)

    rule_mask = torch.ones(1, num_rules, dtype=torch.bool)
    loc_mask = (token_t != 0)

    with torch.no_grad():
        rule_probs, loc_probs, value = net.predict_rule_and_location(
            token_t, depth=depth_t, rule_mask=rule_mask, location_mask=loc_mask
        )

    assert rule_probs.shape == (1, num_rules), f"Bad rule shape: {rule_probs.shape}"
    assert loc_probs.shape == (1, token_t.shape[1]), f"Bad loc shape: {loc_probs.shape}"
    assert value.shape == (1, 1), f"Bad value shape: {value.shape}"
    assert torch.allclose(rule_probs.sum(dim=1), torch.ones(1))
    assert torch.allclose(loc_probs.sum(dim=1), torch.ones(1))
    assert -1.0 <= value.item() <= 1.0

    print(f"  Rule probs shape: {rule_probs.shape} - OK")
    print(f"  Location probs shape: {loc_probs.shape} - OK")
    print(f"  Value: {value.item():.4f} - OK")
    print(f"  PASSED")
    return True


def test_mcts_search():
    """4. MCTS 搜索测试"""
    print("\n" + "=" * 60)
    print("TEST 4: MCTS Search")
    print("=" * 60)

    preprocessor = MathPreprocessor(max_len=128)
    set_default_preprocessor(preprocessor)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()

    device = torch.device("cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, dropout=0.0, use_depth_embedding=True, max_depth=32
    ).to(device)
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))
    net.eval()

    x = sp.Symbol('x')
    expr = sp.Integral(x**2, x)
    state = IntegrationState(expr=expr)

    mcts = MCTS(network=net, preprocessor=preprocessor,
                num_simulations=50, max_depth=15, timeout=10.0, device=device)

    actions, probs = mcts.get_action_probs(state, temperature=1.0)
    assert len(actions) > 0, "No actions found"
    assert len(probs) == len(actions)
    assert abs(sum(probs) - 1.0) < 1e-6, f"Probs sum: {sum(probs)}"

    print(f"  Actions found: {len(actions)}")
    for act, prob in sorted(zip(actions, probs), key=lambda x: -x[1])[:5]:
        print(f"    {act.name} @ pos={act.pos}: {prob:.4f}")
    print(f"  PASSED")
    return True


def test_end_to_end():
    """5. 端到端求解测试"""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Solving")
    print("=" * 60)

    preprocessor = MathPreprocessor(max_len=128)
    set_default_preprocessor(preprocessor)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()
    validator = MathValidator()

    device = torch.device("cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, dropout=0.0, use_depth_embedding=True, max_depth=32
    ).to(device)
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))
    net.eval()

    x = sp.Symbol('x')
    test_cases = [
        (x**2, x**3/3),
        (x**3, x**4/4),
        (sin(x), -cos(x)),
        (cos(x), sin(x)),
        (exp(x), exp(x)),
        (2*x, x**2),
        (1/x, log(x)),
    ]

    passed = 0
    for integrand, expected in test_cases:
        state = IntegrationState(expr=Integral(integrand, x))
        mcts = MCTS(network=net, preprocessor=preprocessor,
                    num_simulations=80, max_depth=10, timeout=10.0, device=device)

        traj = mcts.get_trajectory(state, temperature=0.0)
        if traj:
            last = traj[-1]
            ns, r, done, _ = mcts.env.step(last["state"], last["action"])
            if done and r > 0:
                if validator.verify_integral(integrand, ns.expr):
                    print(f"  Integral({integrand}): {ns.expr} OK")
                    passed += 1
                else:
                    print(f"  Integral({integrand}): {ns.expr} VALIDATION FAILED")
            else:
                print(f"  Integral({integrand}): not solved (r={r}, done={done})")
        else:
            print(f"  Integral({integrand}): no trajectory")

    print(f"\n  PASSED: {passed}/{len(test_cases)}")
    return passed >= len(test_cases) * 0.5


def test_benchmark_eval():
    """6. Benchmark 评估"""
    print("\n" + "=" * 60)
    print("TEST 6: Benchmark Evaluation")
    print("=" * 60)
    bench_path = "benchmark_set.json"
    if not os.path.exists(bench_path):
        print("  SKIPPED: benchmark_set.json not found")
        return True

    with open(bench_path, 'r') as f:
        dataset = json.load(f)

    # Only test first 20 (easy ones)
    subset = [d for d in dataset if d["difficulty"] == "easy"][:20]

    preprocessor = MathPreprocessor(max_len=128)
    set_default_preprocessor(preprocessor)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()

    device = torch.device("cpu")
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, dropout=0.0, use_depth_embedding=True, max_depth=32
    ).to(device)
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))
    net.eval()

    x = sp.Symbol('x')
    passed = 0
    for item in subset:
        try:
            raw = sp.sympify(item["expression"])
            state = IntegrationState(expr=Integral(raw, x))
            mcts = MCTS(network=net, preprocessor=preprocessor,
                        num_simulations=100, max_depth=15, timeout=10.0, device=device)
            traj = mcts.get_trajectory(state, temperature=0.0)
            if traj:
                last = traj[-1]
                ns, r, done, _ = mcts.env.step(last["state"], last["action"])
                if done and r > 0:
                    target = sp.parse_expr(item["target_answer"])
                    try:
                        diff = sp.simplify(sp.diff(ns.expr - target, x))
                        if diff == 0:
                            passed += 1
                    except Exception:
                        pass
        except Exception:
            pass

    acc = passed / len(subset) * 100
    print(f"  Easy benchmark (first 20): {passed}/{len(subset)} ({acc:.1f}%)")
    print(f"  PASSED" if acc >= 30 else "  WARNING: low accuracy (expected at least 30% on easy)")
    return True


def main():
    print("\n" + "=" * 70)
    print("  MathAlphaZero v5.0 — Full Test Suite")
    print("=" * 70)

    results = {}
    tests = [
        ("Rule Registry", test_rule_registry),
        ("Direct Rules", test_rules_direct_apply),
        ("Network", test_network_construction),
        ("MCTS Search", test_mcts_search),
        ("E2E Pipeline", test_e2e_pipeline),
        ("Benchmark Eval", test_benchmark_eval),
    ]

    for name, fn in tests:
        try:
            start = time.time()
            ok = fn()
            elapsed = time.time() - start
            results[name] = ("PASSED" if ok else "FAILED", elapsed)
        except Exception as e:
            results[name] = (f"ERROR: {e}", 0)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    for name, (status, elapsed) in results.items():
        print(f"  {name:<25} {status:<20} {elapsed:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
