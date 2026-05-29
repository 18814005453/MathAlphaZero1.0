# main.py
import os
import torch
import sympy as sp
from sympy import SympifyError

from knowledge.rule_registry import get_all_rule_names, get_num_rules
from core.network import MathNet
from core.engine import MCTS
from utils.preprocessor import MathPreprocessor
from utils.validator import MathValidator
from core.state import IntegrationState


def main():
    print("=========================================")
    print("      MathAlphaZero - 符号积分求解器")
    print("=========================================\n")

    # 定义全局积分变量
    x = sp.Symbol('x')

    # 【优化点 1】自动硬件加速检测 (支持 NVIDIA CUDA, Apple Silicon MPS, 和普通 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 正在初始化计算引擎，当前设备: {device}")

    # 初始化组件
    preprocessor = MathPreprocessor(max_len=128)
    validator = MathValidator()

    # 1. 创建双塔网络
    net = MathNet(
        vocab_size=preprocessor.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        rule_num_layers=2,
        max_len=128,
        dropout=0.1,
        use_depth_embedding=True,
        max_depth=32
    ).to(device)

    # 加载预训练权重
    model_path = "data/brain.pth"
    if os.path.exists(model_path):
        try:
            # 兼容多平台加载
            state_dict = torch.load(model_path, map_location=device)
            # 2. 非严格加载，忽略旧模型中被砍掉的 policy_head
            net.load_state_dict(state_dict, strict=False)
            print("✨ 已成功激活 AI 大脑记忆 (data/brain.pth)\n")
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}，将使用未训练的初始网络。\n")
    else:
        print("⚠️ 未找到训练权重，AI 将采用纯粹的初始状态进行盲搜。\n")

    # 设置网络为评估模式（关闭 Dropout）
    net.eval()

    # ========== 3. 规则缓存注入 ==========
    rule_names = get_all_rule_names()
    action_ids = list(range(get_num_rules()))
    net.refresh_rule_cache(
        rule_texts=rule_names,
        tokenizer_fn=preprocessor.tokenize_list,
        action_ids=action_ids
    )
    print(f"✅ 规则缓存已刷新，当前可用规则数: {len(rule_names)}\n")
    # ==============================================

    # 交互循环
    while True:
        try:
            user_input = input("请输入待积分的表达式 (变量 x, 输入 'q' 退出): ").strip()
            if user_input.lower() in ('q', 'quit', 'exit'):
                print("再见！")
                break
            if not user_input:
                continue

            # 将字符串转换为 SymPy 表达式
            try:
                raw_expr = sp.sympify(user_input)
            except SympifyError:
                print("❌ 表达式解析失败，请检查语法 (例如: x*sin(x), exp(x), 1/(1+x**2))")
                continue

            print(f"\n🤔 正在思考: ∫ {raw_expr} dx ...")

            # 构建初始状态（套上积分符号）
            init_state = IntegrationState(expr=sp.Integral(raw_expr, x))

            # 【优化点 2】每次交互重新实例化 MCTS 时，传入 device 确保张量在同一个设备流转
            mcts = MCTS(
                network=net,
                preprocessor=preprocessor,
                num_simulations=100,
                device=device
            )

            # 执行搜索，temperature=0.0 表示极其贪婪的稳健选择
            trajectory = mcts.get_trajectory(init_state, temperature=0.0)

            success = False
            path = []

            if trajectory:
                last_step = trajectory[-1]
                next_state_raw, reward, done, info = mcts.env.step(last_step["state"], last_step["action"])

                # 收集推理路径
                for step in trajectory:
                    path.append((step["state"].expr, step["action"].name))

                if done and reward > 0:
                    success = True
                    path.append((next_state_raw.expr, "Solved"))

            if not success or not path:
                print("\n❌ 未能找到积分结果。可能原因：")
                print("   - 该积分超出当前规则库范围 (动作空间缺失)")
                print("   - 搜索深度不够（题目过于复杂）")
                print("   - 模型尚未掌握该题型，请运行 train.py 继续进化\n")
                continue

            # 获取最终表达式
            final_expr = path[-1][0]

            # 使用验证器求导确认正确性
            if validator.verify_integral(raw_expr, final_expr):
                print("\n✅ 解题成功！AI 推导步骤如下：")
                for step_idx, (expr, rule_name) in enumerate(path, start=1):
                    if rule_name == "Solved":
                        break
                    print(f"  步骤 {step_idx}: {expr}  ← 应用规则: {rule_name}")
                print(f"\n🎉 最终积分结果: {final_expr} + C\n")
            else:
                print("\n⚠️ 警告：AI 给出的结果未通过求导验证，可能是符号化简盲区或推导错误。")
                print(f"   待验证结果: {final_expr} + C\n")

        except KeyboardInterrupt:
            print("\n\n强制退出。")
            break
        except Exception as e:
            print(f"\n❗ 发生意外错误: {e}")
            continue


if __name__ == "__main__":
    main()