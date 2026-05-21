#!/usr/bin/env python3
# test_solver.py - 快速测试积分求解器

import sys
import os
import sympy
from sympy import Integral, sympify, Symbol

# 添加当前目录到路径
sys.path.insert(0, os.getcwd())

x = Symbol('x')

# 检查核心模块是否可以导入
try:
    from core.rules import RULE_NAMES, RULE_DICT
    print("✓ core.rules 导入成功")
    print(f"  找到 {len(RULE_NAMES)} 条规则: {RULE_NAMES}")
except Exception as e:
    print(f"✗ core.rules 导入失败: {e}")

try:
    from utils.preprocessor import normalize_expression, canonical_integral
    print("✓ utils.preprocessor 导入成功")
except Exception as e:
    print(f"✗ utils.preprocessor 导入失败: {e}")

try:
    from utils.validator import validate_integral
    print("✓ utils.validator 导入成功")
except Exception as e:
    print(f"✗ utils.validator 导入失败: {e}")

try:
    from core.network import IntegratorNet, extract_features
    print("✓ core.network 导入成功")
except Exception as e:
    print(f"✗ core.network 导入失败: {e}")

try:
    from core.engine import mcts_search, apply_rule
    print("✓ core.engine 导入成功")
except Exception as e:
    print(f"✗ core.engine 导入失败: {e}")

# 测试一个简单积分
print("\n" + "="*50)
print("测试简单积分: ∫ x dx")
test_expr = Integral(x, x)
print(f"输入: {test_expr}")

# 尝试使用规则直接求解
from core.rules import rule_power_integral
result = rule_power_integral(test_expr)
print(f"直接应用 PowerIntegral 规则: {result}")

print("\n系统准备就绪！可以运行 python main.py 启动交互式求解器")
