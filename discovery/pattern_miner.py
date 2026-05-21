# discovery/pattern_miner.py
import pickle
from collections import defaultdict, Counter

def load_memory_trajectories(pkl_path):
    """
    加载经验池文件，提取成功轨迹，并转换为适合挖掘的格式。
    期望的 pkl 文件包含字典，键为 "actions", "policy_probs", "q_values" 等。
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: 找不到经验池文件 {pkl_path}")
        return []
    successful_trajs = []
    if isinstance(data, dict):
        actions_list = data.get("actions", [])
        policy_probs_list = data.get("policy_probs", [])
        q_values_list = data.get("q_values", [])
        rewards = data.get("reward", [])
        for i, acts in enumerate(actions_list):
            if rewards[i] > 0:
                traj = {
                    "actions": acts,
                    "policy_probs": policy_probs_list[i] if i < len(policy_probs_list) else [0.0]*len(acts),
                    "q_values": q_values_list[i] if i < len(q_values_list) else [0.0]*len(acts),
                    "reward": rewards[i]
                }
                successful_trajs.append(traj)
    else:
        # 旧格式：列表 of episodes
        for ep in data:
            if ep.get("reward", 0) > 0 or ep.get("success", False):
                successful_trajs.append(ep)
    return successful_trajs

def extract_macro_by_q_delta(trajectories, q_threshold=0.7, policy_threshold=0.1, min_freq=2, top_k=3):
    """
    价值落差挖掘：寻找网络初始策略概率低但最终 Q 值高的连续动作序列。
    返回列表，每个元素为 (action_id1, action_id2) 或 (rule_name1, rule_name2)
    """
    macro_counter = Counter()
    for ep in trajectories:
        actions = ep.get('actions', [])
        policy_probs = ep.get('policy_probs', [])
        q_vals = ep.get('q_values', [])
        if len(actions) < 2 or len(policy_probs) != len(actions) or len(q_vals) != len(actions):
            continue
        # 转换可能为字符串的 action 为 id（如果需要）
        action_ids = []
        for a in actions:
            if isinstance(a, str):
                # 尝试从全局规则名映射 id（需要外部传入，这里先保留字符串）
                action_ids.append(a)
            else:
                action_ids.append(a)
        for i in range(len(action_ids)-1):
            # 检查第一个动作的策略概率低，且第二个动作后的 Q 值高
            if policy_probs[i] < policy_threshold and q_vals[i+1] > q_threshold:
                macro = (action_ids[i], action_ids[i+1])
                macro_counter[macro] += 1
    # 过滤低频
    filtered = {macro: cnt for macro, cnt in macro_counter.items() if cnt >= min_freq}
    sorted_macros = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return [macro for macro, _ in sorted_macros[:top_k]]

def extract_macro_actions(trajectories, n_gram=2, top_k=5, min_freq=2):
    """
    传统的 N-gram 高频挖掘（保留作为备用）
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
            reduction = 1.0
            if has_complexities:
                c_before = complexities[i]
                c_after = complexities[i+n_gram]
                actual_reduction = c_before - c_after
                if actual_reduction > 0:
                    reduction = actual_reduction
                else:
                    continue
            macro_scores[macro] += reduction
            macro_counts[macro] += 1
    valid_macros = {macro: score for macro, score in macro_scores.items() if macro_counts[macro] >= min_freq}
    sorted_macros = sorted(valid_macros.items(), key=lambda x: x[1], reverse=True)
    return [macro for macro, score in sorted_macros[:top_k]]

def map_ids_to_rule_names(macros, rule_names_mapping):
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
            mapped = macro
        readable_macros.append(mapped)
    return readable_macros

# 独立测试
if __name__ == "__main__":
    # 模拟数据
    mock_data = {
        "actions": [[2, 5, 4], [1, 2, 5], [2, 5, 1]],
        "policy_probs": [[0.05, 0.8, 0.1], [0.02, 0.7, 0.2], [0.03, 0.6, 0.3]],
        "q_values": [[0.1, 0.8, 0.9], [0.2, 0.7, 0.95], [0.1, 0.75, 0.88]],
        "reward": [1.0, 1.0, 1.0]
    }
    with open("dummy_miner.pkl", "wb") as f:
        pickle.dump(mock_data, f)
    trajs = load_memory_trajectories("dummy_miner.pkl")
    macros = extract_macro_by_q_delta(trajs, top_k=2, min_freq=2)
    print("价值落差挖掘结果:", macros)
    mock_names = ["Identity", "PowerRule", "TrigProduct", "IntegrationByParts", "ExtractConstant", "LinearSub"]
    readable = map_ids_to_rule_names(macros, mock_names)
    print("可读规则组合:", readable)