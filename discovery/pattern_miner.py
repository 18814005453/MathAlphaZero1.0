# discovery/pattern_miner.py
import pickle
from collections import defaultdict

# 正式对接你真实的知识库
try:
    from knowledge.rules import RULE_NAMES
except ImportError:
    RULE_NAMES = []


def load_memory_trajectories(pkl_path):
    """
    1. RL 经验池解析与清洗
    读取 memory.pkl，过滤出成功拿到正奖励的完美解题轨迹。
    """
    try:
        with open(pkl_path, 'rb') as f:
            memory = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: 找不到经验池文件 {pkl_path}")
        return []

    successful_trajectories = []

    for episode in memory:
        # 兼容不同结构的 reward 记录方式
        reward = episode.get('reward', 0)
        success = episode.get('success', False)

        if reward > 0 or success:
            successful_trajectories.append(episode)

    return successful_trajectories


def extract_macro_actions(trajectories, n_gram=2, top_k=5, min_freq=2):
    """
    2. 高频 N-gram 组合提取 (加固版)
    统计成功路径中的连续动作序列，结合复杂度收益和最小出现频次进行双重筛选。
    """
    macro_scores = defaultdict(float)
    macro_counts = defaultdict(int)

    for episode in trajectories:
        raw_actions = episode.get('actions', [])

        # ✅ 适配点 1：向下兼容。MCTS 如果存的是 Action 对象，自动提取其物理 id
        actions = [a.id if hasattr(a, 'id') else a for a in raw_actions]

        complexities = episode.get('complexities', [])

        # 确保数据长度合法（由于 MCTS 的特殊性，允许 complexities 缺失，退化为仅统计频次）
        has_complexities = (len(complexities) >= len(actions) + 1)

        if len(actions) < n_gram:
            continue

        # 滑动窗口提取 N-gram
        for i in range(len(actions) - n_gram + 1):
            macro = tuple(actions[i: i + n_gram])

            # 默认每次有效推进的基础得分为 1
            reduction = 1.0

            if has_complexities:
                c_before = complexities[i]
                c_after = complexities[i + n_gram]
                # 计算复杂度缩减量（节点数减少的幅度）
                actual_reduction = c_before - c_after
                if actual_reduction > 0:
                    reduction = actual_reduction
                else:
                    # 如果节点数没有减少，视为无效绕路，跳过该组合
                    continue

            macro_scores[macro] += reduction
            macro_counts[macro] += 1

    # ✅ 适配点 2：统计学置信度防御。过滤掉偶然出现（低于 min_freq）的特例组合
    valid_macros = {
        macro: score
        for macro, score in macro_scores.items()
        if macro_counts[macro] >= min_freq
    }

    # 按照综合得分（收益总和）从高到低排序
    sorted_macros = sorted(valid_macros.items(), key=lambda x: x[1], reverse=True)

    # 提取排名前 top_k 的宏动作 ID 组合
    return [macro for macro, score in sorted_macros[:top_k]]


def map_ids_to_rule_names(macro_actions, rule_names_mapping=None):
    """
    3. 动作 ID 到规则文本的翻译
    将连续的物理 ID 翻译回人类可读的字符串规则名，直供 Generator 生成代码。
    """
    if rule_names_mapping is None:
        rule_names_mapping = RULE_NAMES

    readable_macros = []

    for macro in macro_actions:
        try:
            rule_names = [rule_names_mapping[action_id] for action_id in macro]
            readable_macros.append(rule_names)
        except IndexError as e:
            print(f"Warning: 动作 ID {macro} 越界。错误: {e}")
            continue

    return readable_macros


# ==========================================
# 独立测试入口
# ==========================================
if __name__ == "__main__":
    # 使用模拟数据进行脱机测试
    mock_memory_data = [
        {'actions': [2, 5, 4], 'complexities': [20, 18, 10, 5], 'reward': 1.0},
        {'actions': [1, 2, 5], 'complexities': [15, 15, 12, 6], 'reward': 1.0},
        # 新增一条轨迹，满足 min_freq = 2 的阈值要求
        {'actions': [2, 5, 1], 'complexities': [30, 28, 15, 12], 'reward': 1.0},
        {'actions': [1, 1, 1], 'complexities': [10, 10, 10, 10], 'reward': -0.1},
    ]

    with open('dummy_memory.pkl', 'wb') as f:
        pickle.dump(mock_memory_data, f)

    successful_trajs = load_memory_trajectories('dummy_memory.pkl')
    print("1. 过滤出的成功轨迹数:", len(successful_trajs))

    # 开启 min_freq=2，保证挖出来的套路有足够的普适性
    top_macros_ids = extract_macro_actions(successful_trajs, n_gram=2, top_k=2, min_freq=2)
    print("2. 挖掘出的高分宏动作 ID 组合:", top_macros_ids)

    # 如果有真实规则，可以用真实规则测试，否则使用 Mock
    mock_names = ["Identity", "Commutative", "TrigProductToSum", "IntegrationByParts", "PowerRule", "ExtractConstant"]
    final_output = map_ids_to_rule_names(top_macros_ids, rule_names_mapping=mock_names)
    print("3. 输出给 Generator 的最终规则文本对:", final_output)