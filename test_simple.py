import sys
import os
from sympy import Integral, sin, cos, exp, log, sqrt, tan, cot, sec, csc, sinh, cosh, asin, atan

# 确保路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.rules import RULE_DICT
from sympy.abc import x, a


def run_24_table_test():
    # 完美对标标准微积分 24 个基本积分公式
    # 格式：(积分表达式, 预期规则名称, 公式含义)
    test_cases = [
        # 1-5: 幂函数与基本代数
        (Integral(0, x), "PowerIntegral", "0 的积分"),
        (Integral(1, x), "PowerIntegral", "常数 1 的积分"),
        (Integral(x ** a, x), "PowerIntegral", "x^a 的积分 (a != -1)"),
        (Integral(1 / x, x), "PowerIntegral", "1/x 的积分 (ln|x|形式)"),

        # 6-7: 指数函数
        (Integral(exp(x), x), "ExpIntegral", "e^x 的积分"),
        (Integral(a ** x, x), "ExpIntegral", "a^x 的积分"),

        # 8-13: 基本三角函数
        (Integral(sin(x), x), "TrigIntegral", "sin(x) 的积分"),
        (Integral(cos(x), x), "TrigIntegral", "cos(x) 的积分"),
        (Integral(tan(x), x), "TrigIntegral", "tan(x) 的积分"),
        (Integral(cot(x), x), "TrigIntegral", "cot(x) 的积分"),
        (Integral(sec(x), x), "TrigIntegral", "sec(x) 的积分"),
        (Integral(csc(x), x), "TrigIntegral", "csc(x) 的积分"),

        # 14-17: 平方三角函数与乘积
        (Integral(sec(x) ** 2, x), "TrigIntegral", "sec^2(x) 的积分 (tan)"),
        (Integral(csc(x) ** 2, x), "TrigIntegral", "csc^2(x) 的积分 (-cot)"),
        (Integral(sec(x) * tan(x), x), "TrigIntegral", "sec(x)tan(x) 的积分 (sec)"),
        (Integral(csc(x) * cot(x), x), "TrigIntegral", "csc(x)cot(x) 的积分 (-csc)"),

        # 18-19: 双曲函数
        (Integral(sinh(x), x), "HyperbolicIntegral", "sinh(x) 的积分"),
        (Integral(cosh(x), x), "HyperbolicIntegral", "cosh(x) 的积分"),

        # 20-24: 反三角函数源 / 有理无理分式 (最容易报错和崩溃的硬骨头)
        (Integral(1 / (a ** 2 + x ** 2), x), "InvTrigIntegral", "1/(a^2 + x^2) -> arctan"),
        (Integral(1 / sqrt(a ** 2 - x ** 2), x), "InvTrigIntegral", "1/sqrt(a^2 - x^2) -> arcsin"),
        (Integral(1 / (x ** 2 - a ** 2), x), "InvTrigIntegral", "1/(x^2 - a^2) -> ln分式"),
        (Integral(1 / sqrt(x ** 2 + a ** 2), x), "InvTrigIntegral", "1/sqrt(x^2 + a^2) -> ln对数"),
        (Integral(1 / sqrt(x ** 2 - a ** 2), x), "InvTrigIntegral", "1/sqrt(x^2 - a^2) -> ln对数")
    ]

    print("==========================================================")
    print("📋  MathAlphaZero 核心规则库 [标准 24 类基本积分表全面体检]")
    print("==========================================================")

    passed_count = 0
    failed_cases = []

    for idx, (expr, expected_rule, description) in enumerate(test_cases, 1):
        print(f"【No.{idx:02d}】 {description}")
        print(f"       表达式: {expr}")
        matched = False

        for name, rule_func in RULE_DICT.items():
            try:
                result = rule_func(expr)
            except Exception as e:
                print(f"  ❌ 规则 [{name}] 运行崩溃! 错误: {e}")
                continue

            if result is not None:
                next_expr, status = result[0], result[1]
                print(f"  🎯 匹配规则: [{name}] | 状态: [{status}]")
                print(f"  🔄 下步转换: {next_expr}")
                matched = True
                passed_count += 1
                break

        if not matched:
            print("  ❌ 未匹配: 该基础公式未被任何规则捕获！")
            failed_cases.append((description, expr))
        print("-" * 58)

    # 打印体检报告
    print("\n==========================================================")
    print("📊 24 积分表体检报告:")
    print(f"   总测试项: {len(test_cases)} 项")
    print(f"   成功捕获: {passed_count} 项")
    print(f"   覆盖胜率: {(passed_count / len(test_cases)) * 100:.1f}%")

    if failed_cases:
        print(f"   ⚠️ 以下 {len(failed_cases)} 项公式处于盲区，需要补充规则：")
        for desc, f_expr in failed_cases:
            print(f"     - [{desc}]: {f_expr}")
    else:
        print("   🎉 完美的规则库！24个基本积分表全覆盖，地基彻底夯实！")
    print("==========================================================")


if __name__ == "__main__":
    run_24_table_test()

from sympy import Integral, cos, Symbol
from core.rules import rule_split_addition, rule_extract_constant

x = Symbol('x')

# 1. 定义你要测试的积分表达式（赋值给 current_expr）
current_expr = Integral(5 * cos(x), x)
print(f"原始测试题: {current_expr}")

# 2. 设定最大安全步数
max_steps = 10

# 3. 开始循环调度解题
for i in range(max_steps):
    # 找出当前表达式里所有的积分符号
    integrals = list(current_expr.atoms(Integral))
    if not integrals:
        print("🎉 恭喜，所有积分符号已全部消灭！")
        break

    target_integral = integrals[0]

    # 尝试调用你写的规则
    res = rule_split_addition(target_integral)
    if not res:
        res = rule_extract_constant(target_integral)

    if not res:
        # 如果不是运算法则，走原子表秒杀（这里先用 SymPy 的 doit 兜底）
        solved_part = target_integral.doit()
        current_expr = current_expr.subs(target_integral, solved_part)
        print(f"步骤 {i + 1} (查表): 变成 -> {current_expr}")
        continue

    # 拿到拆解或重写后的新积分表达式并替换
    new_sub_expr, status = res
    current_expr = current_expr.subs(target_integral, new_sub_expr)
    print(f"步骤 {i + 1} ({status}): 变成 -> {current_expr}")

print(f"\n最终积分结果: {current_expr}")