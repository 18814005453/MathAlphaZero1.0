import time
import os
import importlib
import torch
import torch.nn as nn

# 假设你的项目模块结构如下：
# import pattern_miner
# import generator
# import knowledge.rules
# from network import DualTowerActorCritic

# ---------------------------------------------------------
# 全局状态跟踪，用于去重防御和命名
# ---------------------------------------------------------
MACRO_HISTORY = set()
NEW_RULE_ID_COUNTER = 1000  # 新规则自增 ID 起点


def run_evolution_loop(net, preprocessor, memory_path="data/memory.pkl", trigger_threshold=500):
    """
    1. 自进化周期调度器 & 2. 管线联动组装
    这是一个独立运行的守护进程，负责监控、冻结、挖掘、生成、重载的完整生命周期。
    """
    print("🚀 [Control Loop] 自进化系统守护进程已启动...")

    while True:
        # 1. 监控触发器：这里用文件大小或更新时间来模拟 trigger
        # 实际业务中可以监听 auto_train.py 发出的 socket 信号，或者轮询 memory.pkl 里的成功记录数
        if not should_trigger_evolution(memory_path, trigger_threshold):
            time.sleep(10)
            continue

        print("\n" + "=" * 50)
        print("🛑 [Control Loop] 触发进化周期！系统控制权移交至进化管线...")

        # 2. 管线联动 - 第一步：挖掘
        trajectories = pattern_miner.load_memory_trajectories(memory_path)
        if not trajectories:
            continue

        # 提取 N-gram (比如2元组) 并翻译成文本
        top_macro_ids = pattern_miner.extract_macro_actions(trajectories, n_gram=2, top_k=3)
        # 获取当前运行时的规则名称列表
        import knowledge.rules
        current_rule_names = knowledge.rules.RULE_NAMES
        top_macros_text = pattern_miner.map_ids_to_rule_names(top_macro_ids, current_rule_names)

        # 2. 管线联动 - 第二步：去重防御与编译
        global NEW_RULE_ID_COUNTER

        for macro_combination in top_macros_text:
            macro_tuple = tuple(macro_combination)

            # 去重防御：如果在过去的周期已经生成过这个连招，跳过
            if macro_tuple in MACRO_HISTORY:
                continue

            print(f"✨ [Pipeline] 挖掘到新的高价值策略套路: {macro_combination}")

            # 交给 generator 生成代码字符串
            rule_name, code_str = generator.generate_macro_rule_code(
                macro_combination,
                new_rule_id=NEW_RULE_ID_COUNTER
            )

            # 改写底层文件
            target_file = "knowledge/rules.py"
            generator.append_rule_to_source_file(target_file, code_str, rule_name)

            # 进行安全验证
            if generator.verify_generated_code(rule_name, file_path=target_file):
                print(f"✅ [Pipeline] {rule_name} 编译验证通过。")

                # 记录历史，递增 ID
                MACRO_HISTORY.add(macro_tuple)
                NEW_RULE_ID_COUNTER += 1

                # 3. 运行时热插拔加载
                hot_reload_knowledge()

                # 4. 双塔网络认知无痛升级
                refresh_brain_cognition(net, preprocessor, rule_name)

            else:
                print(f"❌ [Pipeline] {rule_name} 验证失败，已自动回滚。跳过当前宏。")

        # 进化周期结束，清理或归档当前的 memory.pkl，准备下一次循环
        archive_memory(memory_path)
        print("▶️ [Control Loop] 本轮进化完成，释放控制权，恢复自对弈...")
        print("=" * 50 + "\n")


def hot_reload_knowledge():
    """
    3. 运行时热插拔加载
    强行销毁旧内存，重新加载被 generator 改写后的 rules.py。
    """
    import knowledge.rules

    # Python 内置的 reload 机制，重载模块
    importlib.reload(knowledge.rules)

    new_rule_names = knowledge.rules.RULE_NAMES
    new_rule_dict = knowledge.rules.RULE_DICT

    print(f"🔄 [Hot Reload] 知识库已热重载。当前动作空间维度: {len(new_rule_names)}")
    return new_rule_names, new_rule_dict


def refresh_brain_cognition(net, preprocessor, new_rule_name):
    """
    4. 双塔网络认知无痛升级
    这一步是重构的精华。动作空间从 N 维扩容到 N+1 维。
    因为是双塔网络（状态塔 + 动作规则塔），我们不需要随机初始化新动作的权重，
    而是直接用动作塔（文本编码器）对新生成的规则文本进行特征抽取，实现 “零次学习 (Zero-shot)”。
    """
    import knowledge.rules

    print(f"🧠 [Cognition Refresh] 正在为神经网络扩容新动作: {new_rule_name}...")

    # 步骤 A：更新预处理器字典
    # 预处理器负责把 rule_name 映射为网络输入需要的 Token 或索引
    preprocessor.add_new_rule(new_rule_name)

    with torch.no_grad():
        # 步骤 B：利用动作塔 (Action Tower) 计算新动作的高维语义向量
        # 获取刚热重载进来的新规则的文档字符串 (Docstring) 作为语义来源
        new_rule_func = knowledge.rules.RULE_DICT[new_rule_name]
        rule_description = new_rule_func.__doc__ if new_rule_func.__doc__ else new_rule_name

        # 将文本转换为特征向量 [1, hidden_dim]
        new_action_embedding = net.action_encoder(preprocessor.tokenize(rule_description))

        # 步骤 C：拼接网络权重 (Policy Head 扩容)
        # 假设你的 actor_head 最终计算的是状态向量与动作向量的内积或拼接
        # 如果是常见的分类层 (Linear)，我们将旧矩阵拉长
        if hasattr(net, 'action_embeddings') and isinstance(net.action_embeddings, nn.Parameter):
            old_embeddings = net.action_embeddings.data  # shape: [N, hidden_dim]
            # 拼接到动作词表中 shape: [N+1, hidden_dim]
            expanded_embeddings = torch.cat([old_embeddings, new_action_embedding], dim=0)

            # 强制无痛替换网络参数，不破坏梯度图
            net.action_embeddings = nn.Parameter(expanded_embeddings)

        elif hasattr(net, 'actor_head') and isinstance(net.actor_head, nn.Linear):
            # 如果是全连接层，比如 Linear(state_dim, N)
            old_weight = net.actor_head.weight.data  # shape: [N, state_dim]
            old_bias = net.actor_head.bias.data  # shape: [N]

            state_dim = old_weight.shape[1]
            new_layer = nn.Linear(state_dim, old_weight.shape[0] + 1)

            # 保留原有 N 个动作的肌肉记忆
            new_layer.weight.data[:-1] = old_weight
            new_layer.bias.data[:-1] = old_bias

            # 利用新动作的语义向量初始化最后一行权重（加速收敛）
            # 假设语义向量的维度和 state_dim 一致或可通过投影对齐
            if new_action_embedding.shape[-1] == state_dim:
                new_layer.weight.data[-1] = new_action_embedding.squeeze(0)
            else:
                nn.init.xavier_uniform_(new_layer.weight.data[-1:])

            new_layer.bias.data[-1] = 0.0

            net.actor_head = new_layer

    print(f"⚡ [Cognition Refresh] 升级完成！双塔网络现在可输出 {len(knowledge.rules.RULE_NAMES)} 种动作。")


# ---------------------------------------------------------
# 辅助函数（示意性）
# ---------------------------------------------------------
def should_trigger_evolution(memory_path, threshold):
    """简单模拟触发逻辑：文件存在且达到一定大小则触发"""
    if os.path.exists(memory_path):
        return os.path.getsize(memory_path) > threshold * 10  # 伪逻辑
    return False


def archive_memory(memory_path):
    """归档历史数据，清空当前经验池，准备下个阶段"""
    if os.path.exists(memory_path):
        os.rename(memory_path, memory_path + f".archived_{time.time()}")