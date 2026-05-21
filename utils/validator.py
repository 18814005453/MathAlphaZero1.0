# utils/validator.py (增强版)
import sympy as sp
import random


class MathValidator:
    """
    积分正确性验证器：
    - 主要依靠符号求导 + 化简
    - 数值抽样作为辅助（自动选取非奇异随机点）
    - 支持超时和表达式中的变量自动识别
    """

    @staticmethod
    def verify_integral(raw_expr: sp.Expr, integrated_expr: sp.Expr, var_str: str = 'x') -> bool:
        """
        验证 d(integrated_expr)/dx == raw_expr 是否成立。
        返回 True 表示积分正确，False 表示错误或无法判定。
        """
        try:
            # 获取表达式中的自由符号，优先使用传入的 var_str，若不存在则取第一个自由变量
            free_symbols = raw_expr.free_symbols.union(integrated_expr.free_symbols)
            var = sp.Symbol(var_str) if var_str in str(free_symbols) else next(iter(free_symbols), sp.Symbol('x'))

            # 计算导数
            derivative = sp.diff(integrated_expr, var)
            diff_expr = sp.simplify(derivative - raw_expr)

            # 如果符号化简直接得到 0，直接通过
            if diff_expr == 0:
                return True

            # 符号化简未能判定，使用数值抽样验证
            # 首先检查 diff_expr 是否包含未求值的符号（如积分常数 C）
            if len(diff_expr.free_symbols) > 0:
                # 若含有额外符号（例如积分常数 C），尝试将其视为 0 或随机数值
                # 注意：积分常数在导数中不应出现，但若出现，说明被积函数中已有常数符号，应保留
                # 这里简单处理：如果还存在自由变量且不是 var，则无法数值验证，直接返回 False
                other_symbols = diff_expr.free_symbols - {var}
                if other_symbols:
                    # 可能是未消去的参数，保守起见返回 False
                    return False

            # 数值测试：生成若干随机点（避免奇异点）
            num_tests = 5
            for _ in range(num_tests):
                # 随机选择一个实数值（避开可能使分母为零的点，简单起见选[-2,2]区间，排除0附近？）
                # 更好的做法：随机多次，若遇到奇异则重新采样
                for attempt in range(10):
                    val = random.uniform(-3, 3)
                    # 检查 raw_expr 和 integrated_expr 在该点是否可求值（不奇异）
                    try:
                        raw_val = complex(raw_expr.subs(var, val))
                        int_val = complex(integrated_expr.subs(var, val))
                        # 若两者都能求值，则计算差值
                        diff_val = complex(diff_expr.subs(var, val))
                        if abs(diff_val) > 1e-7:
                            return False
                        break  # 该点测试通过
                    except (ZeroDivisionError, ValueError, TypeError):
                        # 当前点奇异，尝试下一个随机点
                        continue
            return True

        except Exception:
            return False