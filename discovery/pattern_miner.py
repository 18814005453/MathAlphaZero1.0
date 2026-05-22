# discovery/pattern_miner.py
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

def load_memory_trajectories(pkl_path: str) -> List[Dict]:
    """
    加载经验池文件，提取成功轨迹，并转换为适合挖掘的格式。
    期望的 pkl 文件包含字典，键为 "actions", "policy_probs", "q_values", "reward", "complexities" 等。
    若文件不存在或格式不兼容，返回空列表。
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        print(f"Warning: 无法读取经验池文件 {pkl_path}")
        return []

    successful_trajs = []
    if isinstance(data, dict):
        actions_list = data.get("actions", [])
        policy_probs_list = data.get("policy_probs", [])
        q_values_list = data.get("q_values", [])
        rewards = data.get("reward", [])
        complexities = data.get("complexities", [])
        for i, acts in enumerate(actions_list):
            # 只保留成功的轨迹（reward > 0）
            if rewards[i] > 0:
                traj = {
                    "actions": acts,
                    "policy_probs": policy_probs_list[i] if i < len(policy_probs_list) else [0.0] * len(acts),
                    "q_values": q_values_list[i] if i < len(q_values_list) else [0.0] * len(acts),
                    "reward": rewards[i],
                    "complexities": complexities[i] if i < len(complexities) else [0] * (len(acts) + 1)
                }
                successful_trajs.append(traj)
    else:
        # 旧格式：列表 of episodes
        for ep in data:
            if ep.get("reward", 0) > 0 or ep.get("success", False):
                successful_trajs.append(ep)
    return successful_trajs

def extract_macro_by_q_delta(
    trajectories: List[Dict],
    q_threshold: float = 0.7,
    policy_threshold: float = 0.1,
    min_freq: int = 2,
    top_k: int = 3,
    use_complexity_weight: bool = True
) -> List[Tuple]:
    """
    价值落差挖掘：寻找网络初始策略概率低但最终 Q 值高的连续动作序列。
    若 use_complexity_weight 为 True，则根据序列的化简比例加权。
    返回列表，每个元素为 (action_id1, action_id2, ...) 或 (rule_name1, rule_name2, ...)
    """
    macro_counter = defaultdict(float)   # 累计价值（或化简贡献）
    macro_counts = defaultdict(int)

    for ep in trajectories:
        actions = ep.get('actions', [])
        policy_probs = ep.get('policy_probs', [])
        q_vals = ep.get('q_values', [])
        complexities = ep.get('complexities', [])

        if len(actions) < 2 or len(policy_probs) != len(actions) or len(q_vals) != len(actions):
            continue

        # 确保有复杂度序列（长度至少为 actions+1）
        if use_complexity_weight and len(complexities) >= len(actions) + 1:
            has_complexity = True
        else:
            has_complexity = False

        # 将动作转换为 ID（如果是字符串则尝试映射，否则保留原值）
        action_ids = []
        for a in actions:
            if isinstance(a, str):
                # 注意：需要外部提供映射，这里先保留字符串，后续统一处理
                action_ids.append(a)
            else:
                action_ids.append(a)

        # 扫描所有长度为 2 或 3 的连续子序列（可扩展）
        for n in range(2, 4):   # 挖掘 2-gram 和 3-gram
            for i in range(len(action_ids) - n + 1):
                macro = tuple(action_ids[i:i+n])
                # 检查条件：第一个动作的策略概率低，且最后一个动作后的 Q 值高
                if policy_probs[i] < policy_threshold and q_vals[i+n-1] > q_threshold:
                    weight = 1.0
                    if has_complexity:
                        # 计算该子序列造成的复杂度减少比例
                        c_before = complexities[i]
                        c_after = complexities[i+n]
                        reduction = (c_before - c_after) / max(c_before, 1)
                        # 加权：化简越多，权重越高
                        weight = 1.0 + reduction
                    macro_counter[macro] += weight
                    macro_counts[macro] += 1

    # 过滤低频
    filtered = {macro: score for macro, score in macro_counter.items() if macro_counts[macro] >= min_freq}
    sorted_macros = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return [macro for macro, _ in sorted_macros[:top_k]]

def extract_macro_actions(
    trajectories: List[Dict],
    n_gram: int = 2,
    top_k: int = 5,
    min_freq: int = 2,
    use_complexity_weight: bool = True
) -> List[Tuple]:
    """
    传统的 N-gram 高频挖掘（备用），支持根据化简比例加权。
    """
    macro_scores = defaultdict(float)
    macro_counts = defaultdict(int)
    for ep in trajectories:
        raw_actions = ep.get('actions', [])
        actions = [a.id if hasattr(a, 'id') else a for a in raw_actions]
        complexities = ep.get('complexities', [])
        has_complexities = (len(complexities) >= len(actions) + 1)
        if len(actions) < n_gram:
            continue
        for i in range(len(actions) - n_gram + 1):
            macro = tuple(actions[i:i+n_gram])
            weight = 1.0
            if use_complexity_weight and has_complexities:
                c_before = complexities[i]
                c_after = complexities[i+n_gram]
                reduction = (c_before - c_after) / max(c_before, 1)
                if reduction > 0:
                    weight = 1.0 + reduction
                else:
                    continue   # 无化简则跳过
            macro_scores[macro] += weight
            macro_counts[macro] += 1
    valid_macros = {macro: score for macro, score in macro_scores.items() if macro_counts[macro] >= min_freq}
    sorted_macros = sorted(valid_macros.items(), key=lambda x: x[1], reverse=True)
    return [macro for macro, _ in sorted_macros[:top_k]]

def map_ids_to_rule_names(
    macros: List[Tuple],
    rule_names_mapping: List[str] or Dict[int, str]
) -> List[List[str]]:
    """
    将动作 ID 序列映射为规则名字符串序列。
    rule_names_mapping: 可以是 list (索引->名称) 或 dict (id->名称)
    """
    readable_macros = []
    for macro in macros:
        if isinstance(macro[0], int):
            if isinstance(rule_names_mapping, list):
                mapped = [rule_names_mapping[i] for i in macro]
            else:
                mapped = [rule_names_mapping.get(i, str(i)) for i in macro]
        else:
            # 已经是名字
            mapped = list(macro)
        readable_macros.append(mapped)
    return readable_macros

def filter_macros_by_simplification(
    macros: List[Tuple],
    trajectories: List[Dict],
    min_reduction_ratio: float = 0.1
) -> List[Tuple]:
    """
    进一步筛选宏规则，只保留那些在实际轨迹中平均化简比例超过阈值的。
    """
    macro_reduction = defaultdict(list)
    for ep in trajectories:
        actions = ep.get('actions', [])
        complexities = ep.get('complexities', [])
        if len(complexities) < len(actions) + 1:
            continue
        # 将动作转换为可比较形式（字符串或整数）
        acts = [a.id if hasattr(a, 'id') else a for a in actions]
        for macro in macros:
            macro_len = len(macro)
            for i in range(len(acts) - macro_len + 1):
                if tuple(acts[i:i+macro_len]) == macro:
                    c_before = complexities[i]
                    c_after = complexities[i+macro_len]
                    reduction = (c_before - c_after) / max(c_before, 1)
                    macro_reduction[macro].append(reduction)
    # 计算平均化简比例
    filtered = []
    for macro, reds in macro_reduction.items():
        avg_red = sum(reds) / len(reds)
        if avg_red >= min_reduction_ratio:
            filtered.append(macro)
    return filtered

# 独立测试
if __name__ == "__main__":
    # 模拟数据
    mock_data = {
        "actions": [[2, 5, 4], [1, 2, 5], [2, 5, 1]],
        "policy_probs": [[0.05, 0.8, 0.1], [0.02, 0.7, 0.2], [0.03, 0.6, 0.3]],
        "q_values": [[0.1, 0.8, 0.9], [0.2, 0.7, 0.95], [0.1, 0.75, 0.88]],
        "reward": [1.0, 1.0, 1.0],
        "complexities": [[10, 9, 7, 5], [12, 10, 8, 6], [15, 13, 10, 9]]
    }
    with open("dummy_miner.pkl", "wb") as f:
        pickle.dump(mock_data, f)
    trajs = load_memory_trajectories("dummy_miner.pkl")
    macros = extract_macro_by_q_delta(trajs, top_k=2, min_freq=2, use_complexity_weight=True)
    print("价值落差挖掘结果:", macros)
    mock_names = ["Identity", "PowerRule", "TrigProduct", "IntegrationByParts", "ExtractConstant", "LinearSub"]
    readable = map_ids_to_rule_names(macros, mock_names)
    print("可读规则组合:", readable)
    # 过滤测试
    filtered = filter_macros_by_simplification(macros, trajs, min_reduction_ratio=0.1)
    print("过滤后:", filtered)