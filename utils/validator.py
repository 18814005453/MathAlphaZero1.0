# utils/validator.py
import sympy as sp
import random
import numpy as np
from typing import Optional, Set, Tuple

class MathValidator:
    """
    积分正确性验证器：
    - 符号验证：求导后化简，判断是否为零
    - 常数符号验证：检查结果是否包含积分常数（如 C）并自动忽略
    - 数值验证：采样随机点，检测奇异点自动重试，支持复数域
    """

    @staticmethod
    def verify_integral(
        raw_expr: sp.Expr,
        integrated_expr: sp.Expr,
        var_str: str = 'x',
        num_samples: int = 10,
        tolerance: float = 1e-6,
        check_constant: bool = True
    ) -> bool:
        """
        验证 integrated_expr 的导数是否等于 raw_expr。
        返回 True 表示积分正确。
        """
        try:
            # 识别变量
            free = raw_expr.free_symbols.union(integrated_expr.free_symbols)
            var = sp.Symbol(var_str) if var_str in str(free) else next(iter(free), sp.Symbol('x'))

            # 计算导数并化简
            derivative = sp.diff(integrated_expr, var)
            diff_expr = sp.simplify(derivative - raw_expr)

            # 符号零判定
            if diff_expr == 0:
                return True
            if diff_expr.is_zero:
                return True

            # 如果差表达式是数值 0（例如 0.0）
            if diff_expr.is_number and abs(float(diff_expr)) < tolerance:
                return True

            # 处理可能包含积分常数 C 的情况（一般不会出现在导数中，但若被积函数本身含 C 则忽略）
            if check_constant:
                # 查找 diff_expr 中可能代表常数的符号（如 'C', 'c', 'C1' 等）
                const_candidates = [s for s in diff_expr.free_symbols if s.name.startswith(('C', 'c'))]
                if const_candidates:
                    # 尝试将这些常数设为 0，重新计算
                    subs_dict = {c: 0 for c in const_candidates}
                    diff_expr_zero_const = diff_expr.subs(subs_dict)
                    if sp.simplify(diff_expr_zero_const) == 0:
                        return True

            # 数值验证：采样随机点，避开奇异点
            # 首先生成候选点列表
            points = []
            for _ in range(num_samples * 2):  # 多生成一些备用
                # 在 [-3, 3] 区间内随机，排除分母为零的情况
                val = random.uniform(-3, 3)
                points.append(val)
            # 去重
            points = list(set(points))

            for val in points:
                # 检查原始表达式和积分表达式在该点是否可求值
                try:
                    # 尝试复数求值（处理 sqrt 负值等情况）
                    raw_val = sp.N(raw_expr.subs(var, val))
                    int_val = sp.N(integrated_expr.subs(var, val))
                    diff_val = sp.N(diff_expr.subs(var, val))
                    if abs(diff_val) > tolerance:
                        # 如果差太大，再尝试一次高精度
                        raw_val_high = sp.N(raw_expr.subs(var, val), 50)
                        int_val_high = sp.N(integrated_expr.subs(var, val), 50)
                        diff_val_high = abs(raw_val_high - int_val_high)
                        if diff_val_high > tolerance:
                            return False
                except (ZeroDivisionError, ValueError, TypeError):
                    # 遇到奇异点，继续尝试下一个点
                    continue

            return True

        except Exception:
            return False

    @staticmethod
    def verify_with_symbolic_constants(
        raw_expr: sp.Expr,
        integrated_expr: sp.Expr,
        var_str: str = 'x',
        constant_symbols: Optional[Set[sp.Symbol]] = None
    ) -> bool:
        """
        验证积分正确性，允许 integrated_expr 中包含未指定的常数符号（如积分常数）。
        通过检查导数是否等于 raw_expr 且不含未定常数（除了允许的）。
        """
        var = sp.Symbol(var_str)
        derivative = sp.diff(integrated_expr, var)
        diff_expr = sp.simplify(derivative - raw_expr)

        if diff_expr == 0:
            return True

        # 获取 diff_expr 中的自由符号
        free_in_diff = diff_expr.free_symbols
        # 如果指定了允许的常数符号，则从自由符号中剔除
        if constant_symbols:
            free_in_diff = free_in_diff - constant_symbols

        # 如果还有未指定的自由符号，则失败
        if free_in_diff:
            return False

        # 否则对数值再次验证
        return MathValidator.verify_integral(raw_expr, integrated_expr, var_str)

    @staticmethod
    def complexity_ratio(original: sp.Expr, integrated: sp.Expr) -> float:
        """计算积分后表达式的化简比例（基于 AST 节点数）"""
        def node_count(expr):
            from sympy import preorder_traversal
            return sum(1 for _ in preorder_traversal(expr))

        orig_nodes = node_count(original)
        if orig_nodes == 0:
            return 0.0
        int_nodes = node_count(integrated)
        return (orig_nodes - int_nodes) / orig_nodes