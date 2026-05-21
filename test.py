# test_simple.py
from knowledge.rules import RULE_DICT
from sympy import Integral
from sympy.abc import x

print("开始测试...")
expr = Integral(x**2, x)

for name, rule_func in RULE_DICT.items():
    result = rule_func(expr)
    if result is not None:
        print(f"✅ 匹配成功! 规则名: {name}, 结果: {result[0]}")
        break
else:
    print("❌ 未匹配到任何规则")