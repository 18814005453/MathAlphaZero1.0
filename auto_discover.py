# auto_discover.py
import time
import os
import pickle
import importlib
import torch
from collections import defaultdict

import discovery.pattern_miner as pattern_miner
import discovery.generator as generator
from knowledge.rule_registry import get_all_rule_names, build_action_space, reload_module, get_num_rules
from core.network import MathNet
from utils.preprocessor import MathPreprocessor

# 全局历史记录，防止重复生成相同宏
MACRO_HISTORY = set()
NEW_RULE_ID_COUNTER = 1000

def should_trigger_evolution(memory_path, threshold):
    """判断是否触发进化：积累足够的新成功轨迹"""
    if os.path.exists(memory_path) and os.path.getsize(memory_path) > 0:
        try:
            with open(memory_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "actions" in data:
                # 成功轨迹数量达到阈值
                return len(data["actions"]) >= threshold
        except Exception:
            return False
    return False

def hot_reload_knowledge():
    """热重载 knowledge.rules 并重建动作空间"""
    reload_module("knowledge.rules")
    new_rule_names = get_all_rule_names()
    print(f"🔄 热重载完成，当前动作空间大小: {len(new_rule_names)}")
    return new_rule_names

def refresh_brain_cognition(net, preprocessor, new_rule_name=None):
    """更新神经网络规则缓存"""
    from knowledge.rule_registry import get_all_rule_names, get_num_rules
    rule_names = get_all_rule_names()
    num_rules = get_num_rules()
    # 重新生成动作 ID 映射（保持 ID 连续）
    action_ids = list(range(num_rules))
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids=action_ids)
    # 如果网络有输出维度调整方法，调用它（可选）
    if hasattr(net, 'adjust_output_dim'):
        net.adjust_output_dim(num_rules)
    print(f"🧠 神经网络规则缓存已刷新，输出维度: {num_rules}")

def run_evolution_loop(net, preprocessor, memory_path="data/memory_final_for_miner.pkl", trigger_threshold=10):
    """
    自进化守护进程：监控经验池，挖掘高价值宏规则，动态生成并热加载。
    """
    global NEW_RULE_ID_COUNTER, MACRO_HISTORY

    print("🚀 [自进化] 守护进程已启动，监控路径:", memory_path)

    while True:
        # 等待触发条件
        if not should_trigger_evolution(memory_path, trigger_threshold):
            time.sleep(30)
            continue

        print("\n" + "=" * 60)
        print("🛑 触发进化周期，开始挖掘高价值动作序列...")

        # 1. 加载经验池数据
        try:
            with open(memory_path, 'rb') as f:
                miner_data = pickle.load(f)
        except Exception as e:
            print(f"⚠️ 无法读取经验池: {e}")
            time.sleep(60)
            continue

        # 2. 转换为轨迹列表格式（pattern_miner 期望的格式）
        trajectories = []
        if isinstance(miner_data, dict) and "actions" in miner_data:
            for i in range(len(miner_data["actions"])):
                traj = {
                    "actions": miner_data["actions"][i],
                    "policy_probs": miner_data.get("policy_probs", [[]])[i] if i < len(miner_data.get("policy_probs", [])) else [],
                    "q_values": miner_data.get("q_values", [[]])[i] if i < len(miner_data.get("q_values", [])) else [],
                    "reward": miner_data["reward"][i] if i < len(miner_data["reward"]) else 0,
                    "complexities": miner_data.get("complexities", [[]])[i] if i < len(miner_data.get("complexities", [])) else []
                }
                if traj["reward"] > 0:
                    trajectories.append(traj)
        else:
            print("⚠️ 经验池格式不兼容，跳过此轮")
            time.sleep(60)
            continue

        if len(trajectories) < 2:
            print("ℹ️ 成功轨迹不足，等待更多数据")
            time.sleep(60)
            continue

        # 3. 价值落差挖掘（优先）
        top_macros_ids = pattern_miner.extract_macro_by_q_delta(
            trajectories,
            q_threshold=0.7,
            policy_threshold=0.1,
            min_freq=2,
            top_k=3,
            use_complexity_weight=True
        )
        if not top_macros_ids:
            # 备用：传统 N-gram 挖掘
            top_macros_ids = pattern_miner.extract_macro_actions(
                trajectories, n_gram=2, top_k=2, min_freq=2, use_complexity_weight=True
            )

        if not top_macros_ids:
            print("❌ 未挖掘到有效宏模式，跳过本轮")
            time.sleep(60)
            continue

        # 4. 将 ID 序列翻译为规则名称
        current_rule_names = get_all_rule_names()
        # 建立 ID->名称映射（假设动作 ID 为整数索引）
        id_to_name = {idx: name for idx, name in enumerate(current_rule_names)}
        readable_macros = pattern_miner.map_ids_to_rule_names(top_macros_ids, id_to_name)

        # 5. 去重并生成代码
        for macro_names in readable_macros:
            macro_tuple = tuple(macro_names)
            if macro_tuple in MACRO_HISTORY:
                print(f"⏩ 宏 {macro_names} 已存在，跳过")
                continue

            print(f"✨ 发现新宏模式: {macro_names}")
            rule_name, code_str = generator.generate_macro_rule_code(macro_names, NEW_RULE_ID_COUNTER)

            target_file = "knowledge/rules.py"
            try:
                generator.append_rule_to_source_file(target_file, code_str, rule_name)
                # 验证新规则
                if generator.verify_generated_code(rule_name, target_file):
                    # 热重载知识库
                    hot_reload_knowledge()
                    # 更新神经网络缓存
                    refresh_brain_cognition(net, preprocessor, rule_name)
                    MACRO_HISTORY.add(macro_tuple)
                    NEW_RULE_ID_COUNTER += 1
                    # 创建信号文件，通知训练进程更新
                    with open("data/RELOAD_FLAG", "w") as f:
                        f.write(rule_name)
                    print(f"✅ 宏规则 {rule_name} 已成功注入系统")
                else:
                    print(f"❌ 宏规则 {rule_name} 验证失败，已回滚")
            except Exception as e:
                print(f"❌ 生成宏规则时出错: {e}")

        # 6. 清理低频宏（每轮进化后执行一次）
        generator.prune_inactive_macros(threshold=30, file_path="knowledge/rules.py")

        # 7. 重置或归档当前经验池（避免重复挖掘）
        if os.path.exists(memory_path):
            backup_path = memory_path + f".archived_{int(time.time())}"
            os.rename(memory_path, backup_path)
            print(f"📦 已归档当前经验池至 {backup_path}")

        print("▶️ 本轮进化完成，等待下一轮触发...")
        print("=" * 60 + "\n")
        time.sleep(10)

# 独立运行测试
if __name__ == "__main__":
    # 导入必要的组件（假设外部存在）
    from core.network import MathNet
    from utils.preprocessor import MathPreprocessor
    from knowledge.rule_registry import get_all_rule_names

    # 初始化预处理器和网络（用于占位）
    preprocessor = MathPreprocessor(max_len=128)
    rule_names = get_all_rule_names()
    num_actions = len(rule_names)
    net = MathNet(vocab_size=preprocessor.vocab_size, num_actions=num_actions)
    # 初始化规则缓存
    action_ids = list(range(num_actions))
    net.refresh_rule_cache(rule_names, preprocessor._string_to_ids, action_ids=action_ids)

    # 启动进化循环
    run_evolution_loop(net, preprocessor, memory_path="data/memory_final_for_miner.pkl", trigger_threshold=5)