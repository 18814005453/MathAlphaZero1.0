# main.py
import os
import sys
import torch
import sympy as sp
from sympy import SympifyError

from core.rules import MathRuleBase
from core.network import MathAlphaZeroNet
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

    # 初始化组件
    preprocessor = MathPreprocessor(max_len=128)
    rules = MathRuleBase()
    validator = MathValidator()

    # 创建网络
    net = MathAlphaZeroNet(
        vocab_size=preprocessor.vocab_size,
        num_actions=rules.num_actions,
        d_model=128,
        nhead=4,
        num_layers=3
    )

    # 加载预训练权重
    model_path = "data/brain.pth"
    if os.path.exists(model_path):
        try:
            # 兼容 Mac CPU 运行
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            print("✨ 已成功激活 AI 大脑记忆 (data/brain.pth)\n")
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}，将使用未训练的初始网络。\n")
    else:
        print("⚠️ 未找到训练权重，AI 将采用纯粹的初始状态进行盲搜。\n")

    # 设置网络为评估模式（关闭 Dropout 等，保证推理稳定性）
    net.eval()

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

            # 🌟 核心修复：显式套上积分符号，对齐 env.py 的匹配逻辑
            init_state = IntegrationState(expr=sp.Integral(raw_expr, x))

            # 每次交互重新实例化 MCTS 清空树缓存，给予足够深的搜索次数
            mcts = MCTS(network=net, preprocessor=preprocessor, num_simulations=100)

            # 执行搜索，temperature=0.0 表示在实际解题时进行极其贪婪的稳健选择
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
                print("   - 模型尚未掌握该题型，请运行 auto_train.py 继续进化\n")
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