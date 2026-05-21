import time
import os
import importlib
import torch
import torch.nn as nn

# 假设你的项目模块结构如下：
import discovery.pattern_miner as pattern_miner
import discovery.generator as generator
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

        # 2. 管线联动 - 第一步：挖掘与数据适配重构
        import pickle
        trajectories = []
        if os.path.exists(memory_path):
            try:
                with open(memory_path, 'rb') as f_miner_data:
                    raw_miner_dict = pickle.load(f_miner_data)
                if isinstance(raw_miner_dict, dict) and "actions" in raw_miner_dict:
                    # 将并行的 actions/reward 列表重新拼装为 Episode 字典字典列表
                    for acts, r in zip(raw_miner_dict["actions"], raw_miner_dict["reward"]):
                        trajectories.append({"actions": acts, "reward": r})
            except Exception as e:
                print(f"⚠️ 读取挖掘池失败: {e}")
                
        if not trajectories:
            print("ℹ️ 暂无合规的成功解题轨迹可供挖掘，跳过本次周期")
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
            
            try:
                # 3. 强行触发运行时热插拔加载与语法检查
                hot_reload_knowledge()
                
                print(f"🚀 [Pipeline] {rule_name} 物理写入并重载成功！强行锁定进化状态。")
                MACRO_HISTORY.add(macro_tuple)
                NEW_RULE_ID_COUNTER += 1
                
                # 创建一个强有力的物理信号文件，显式通知主训练进程更新缓存
                with open("data/RELOAD_FLAG", "w") as f_flag:
                    f_flag.write(rule_name)

                # 4. 双塔网络认知无痛升级
                refresh_brain_cognition(net, preprocessor, rule_name)
            except Exception as e:
                print(f"❌ [Pipeline] {rule_name} 写入导致语法崩溃，执行安全回滚: {e}")
                if hasattr(generator, 'rollback_source_file'):
                    generator.rollback_source_file(target_file)

        # 进化周期结束，清理或归档当前的 memory.pkl，准备下一次循环
        print("📊 [Control Loop] 保留当前轨迹池，等待累积更多成功样本...")
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
    4. 双塔网络认知无痛升级（修改后：利用 refresh_rule_cache 重新编码规则）
    """
    import knowledge.rules

    print(f"🧠 [Cognition Refresh] 正在为神经网络扩容新动作: {new_rule_name}")

    # 步骤 A：更新预处理器的规则映射（如果预处理器支持）
    if hasattr(preprocessor, 'add_new_rule'):
        preprocessor.add_new_rule(new_rule_name)

    # 步骤 B：刷新网络的规则缓存（双塔架构会自动为新规则生成语义向量）
    # 注意：此处必须传入一个能够将规则文本转换为 token 序列的函数
    # 假设 preprocessor 有一个 tokenize_rule 方法，若没有则需自定义
    if hasattr(preprocessor, 'tokenize_rule'):
        tokenizer_fn = preprocessor.tokenize_rule
    elif hasattr(preprocessor, 'tokenize'):
        tokenizer_fn = preprocessor.tokenize
    else:
        # 后备方案：使用预处理器中已有的 state_to_tensor（但可能不适用于纯文本）
        # 实际项目中推荐实现专门的规则 tokenizer
        tokenizer_fn = lambda s: preprocessor.state_to_tensor(s)  # 风险：可能格式错误
        print("⚠️ 警告：预处理器缺少规则专用 tokenizer，使用 state_to_tensor 替代，可能导致编码错误。")

    net.refresh_rule_cache(
        knowledge.rules.RULE_NAMES,
        tokenizer_fn=tokenizer_fn
    )
    print(f"⚡ [Cognition Refresh] 升级完成！双塔网络现在可输出 {len(knowledge.rules.RULE_NAMES)} 种动作。")


# ---------------------------------------------------------
# 辅助函数（示意性）
# ---------------------------------------------------------
def should_trigger_evolution(memory_path, threshold):
    """硬核改版：直接读取成功题数，达到设定值立刻触发进化"""
    import pickle
    if os.path.exists(memory_path) and os.path.getsize(memory_path) > 0:
        try:
            with open(memory_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "actions" in data:
                # 只要累积成功解出的题目达到 10 道（即 10 条成功轨迹），就激活宏挖掘
                return len(data["actions"]) >= 10
        except Exception:
            return False
    return False


def archive_memory(memory_path):
    """归档历史数据，清空当前经验池，准备下个阶段"""
    if os.path.exists(memory_path):
        os.rename(memory_path, memory_path + f".archived_{time.time()}")
if __name__ == "__main__":
    from core.network import MathNet
    from utils.preprocessor import MathPreprocessor
    
    # 实例化最基础的组件，用来作为热加载的句柄
    preprocessor = MathPreprocessor(max_len=128)
    net = MathNet(vocab_size=preprocessor.vocab_size)
    
    # 启动进化守护进程，将触发阈值设为 300 条轨迹记录
    run_evolution_loop(
        net=net, 
        preprocessor=preprocessor, 
        memory_path="data/memory_final_for_miner.pkl", 
        trigger_threshold=300
    )
