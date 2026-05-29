#!/usr/bin/env python3
"""
MathAlphaZero 自动规则发现引擎 v5.0

完整流程：
1. 从经验池挖掘高频成功动作序列 (pattern_miner)
2. 生成组合宏规则代码 (generator)
3. AST 安全注入到 knowledge/rules.py
4. 验证通过后热重载 -> 网络缓存刷新
5. 训练进程检测 RELOAD_FLAG -> 自动更新
"""

import os
import sys
import time
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discovery.pattern_miner import (
    load_memory_trajectories, extract_macro_by_q_delta,
    extract_macro_actions, map_ids_to_rule_names
)
from discovery.generator import (
    generate_macro_rule_code, append_rule_to_rules_file,
    verify_generated_code, prune_inactive_macros, macro_usage_counter
)
from knowledge.rule_registry import (
    get_all_rule_names, build_action_space, reload_module, get_num_rules
)

MACRO_HISTORY = set()
NEW_RULE_ID_COUNTER = 1000


def discover_from_memory(memory_path: str, net=None, preprocessor=None, top_k=3):
    """Run one discovery cycle on the given experience buffer."""
    global NEW_RULE_ID_COUNTER, MACRO_HISTORY

    if not os.path.exists(memory_path) or os.path.getsize(memory_path) == 0:
        return []

    # Load experience
    try:
        with open(memory_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Cannot read memory: {e}")
        return []

    # Convert to trajectory format
    trajectories = []
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        buffer, priorities, pos = data
        for item in buffer:
            # Each item is (token_tensor, depth_tensor, rule_target, loc_target, value_target)
            trajectories.append({
                "reward": float(item[4].item() if hasattr(item[4], 'item') else item[4]),
                "actions": [],
                "policy_probs": [],
                "q_values": [],
            })
    elif isinstance(data, dict) and "actions" in data:
        for i in range(len(data["actions"])):
            traj = {
                "actions": data["actions"][i],
                "policy_probs": data.get("policy_probs", [[]])[i] if i < len(data.get("policy_probs", [])) else [],
                "q_values": data.get("q_values", [[]])[i] if i < len(data.get("q_values", [])) else [],
                "reward": data["reward"][i] if i < len(data["reward"]) else 0,
                "complexities": data.get("complexities", [[]])[i] if i < len(data.get("complexities", [])) else [],
            }
            if traj["reward"] > 0:
                trajectories.append(traj)

    successful = [t for t in trajectories if t.get("reward", 0) > 0]
    if len(successful) < 2:
        print(f"Not enough successful trajectories ({len(successful)}), need >= 2")
        return []

    print(f"Analyzing {len(successful)} successful trajectories...")

    # Try Q-delta mining first, then N-gram
    top_macros = extract_macro_by_q_delta(
        successful, q_threshold=0.5, policy_threshold=0.2,
        min_freq=2, top_k=top_k, use_complexity_weight=True
    )

    if not top_macros:
        top_macros = extract_macro_actions(
            successful, n_gram=2, top_k=top_k, min_freq=2, use_complexity_weight=True
        )

    if not top_macros:
        print("No macro patterns discovered in this cycle")
        return []

    # Map IDs to rule names
    id_to_name = {idx: name for idx, name in enumerate(get_all_rule_names())}
    readable = map_ids_to_rule_names(top_macros, id_to_name)

    new_rules = []
    for macro_names in readable:
        macro_tuple = tuple(macro_names)
        if macro_tuple in MACRO_HISTORY:
            print(f"  Skip duplicate: {macro_names}")
            continue

        print(f"  New pattern: {macro_names}")
        rule_name, code_str = generate_macro_rule_code(list(macro_names), NEW_RULE_ID_COUNTER)

        target = "knowledge/rules.py"
        try:
            append_rule_to_rules_file(target, code_str, rule_name)
            if verify_generated_code(rule_name, target):
                reload_module("knowledge.rules")
                if net is not None and preprocessor is not None:
                    names = get_all_rule_names()
                    net.refresh_rule_cache(names, preprocessor._string_to_ids,
                                           action_ids=list(range(get_num_rules())))
                MACRO_HISTORY.add(macro_tuple)
                NEW_RULE_ID_COUNTER += 1
                new_rules.append(rule_name)

                # Signal training process
                with open("data/RELOAD_FLAG", "w") as f:
                    f.write(rule_name)
                print(f"  NEW RULE ADDED: {rule_name} (total: {get_num_rules()})")
            else:
                print(f"  Verification failed for {rule_name}, rolled back")
        except Exception as e:
            print(f"  Error: {e}")

    # Prune inactive macros
    prune_inactive_macros(threshold=30, file_path="knowledge/rules.py")

    return new_rules


def run_discovery_once(memory_path="data/memory.pkl", net=None, preprocessor=None):
    """Single discovery cycle — call from training loop."""
    return discover_from_memory(memory_path, net, preprocessor, top_k=3)


if __name__ == "__main__":
    import torch
    from core.network import MathNet
    from utils.preprocessor import MathPreprocessor
    from knowledge.rule_registry import build_action_space
    import knowledge.rules

    build_action_space()
    print(f"Initial rules: {get_num_rules()}")

    preprocessor = MathPreprocessor(max_len=128)
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()

    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=64, nhead=4, num_layers=2, rule_num_layers=2,
        max_len=128, use_depth_embedding=True, max_depth=32
    )
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids,
                           action_ids=list(range(num_rules)))

    # Demo: create a synthetic memory and test discovery
    print("\n--- Discovery Demo ---")
    demo_rules = discover_from_memory("data/memory.pkl", net, preprocessor)
    print(f"\nDiscovered {len(demo_rules)} new rules: {demo_rules}")
    print(f"Final rule count: {get_num_rules()}")
