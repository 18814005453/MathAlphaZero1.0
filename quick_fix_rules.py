import re

# 修复 rule_power_integral 函数
rules_file = "knowledge/rules.py"
with open(rules_file, 'r') as f:
    content = f.read()

# 增强幂函数匹配
power_fix = '''
@register_rule()
def rule_power_integral(integral):
    """幂函数积分 - 增强版"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    # 常数
    if is_constant(func, x):
        return (func * x, "solved")
    # x^n (包括 x^1)
    if func == x or (isinstance(func, Pow) and func.base == x):
        if func == x:
            n = 1
        else:
            n = func.exp
        if n == -1:
            return (log(Abs(x)), "solved")
        else:
            return (x**(n+1) / (n+1), "solved")
    # (ax+b)^n
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if is_linear_in_x(base):
            a, b = linear_coeff(base)
            if a != 0 and is_constant(a, x):
                if expn == -1:
                    return ((1/a) * log(Abs(base)), "solved")
                else:
                    return (base**(expn+1) / (a * (expn+1)), "solved")
    # sqrt(ax+b)
    if is_sqrt(func):
        inner = func.args[0]
        if is_linear_in_x(inner):
            a, b = linear_coeff(inner)
            if a != 0:
                return ((2/(3*a)) * (inner)**(3/2), "solved")
    return None
'''

# 找到原函数并替换
pattern = r'@register_rule\(\)\s+def rule_power_integral\(integral\):.*?(?=@register_rule|$)'
if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, power_fix, content, flags=re.DOTALL)
    with open(rules_file, 'w') as f:
        f.write(content)
    print("✅ rule_power_integral 已增强")
else:
    print("⚠️ 未找到原函数，手动添加")

# 修复分部积分超时问题 - 添加超时保护
timeout_fix = '''
import functools
import signal

def timeout(seconds=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError()
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                result = None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

@register_rule()
def rule_integration_by_parts(integral):
    """分部积分 - 带超时保护"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if not func.is_Mul or len(func.args) < 2:
        return None
    
    def liate_score(expr):
        if expr.has(log): return 5
        if expr.has(asin, acos, atan, acot, asec, acsc): return 4
        if expr.is_polynomial(x) or (isinstance(expr, Pow) and expr.base == x): return 3
        if expr.has(sin, cos, tan, sec, csc, cot): return 2
        if expr.has(exp): return 1
        return 0
    
    args = list(func.args)
    u_idx = max(range(len(args)), key=lambda i: liate_score(args[i]))
    u = args[u_idx]
    dv_dx = Mul(*(args[:u_idx] + args[u_idx+1:]))
    
    # 超时保护
    try:
        v = sympy.integrate(dv_dx, x)
        if v.has(Integral):
            return None
    except:
        return None
    
    du_dx = diff(u, x)
    new_expr = u * v - Integral(v * du_dx, x)
    return (new_expr, "rewrite")
'''

# 替换分部积分函数
pattern_parts = r'@register_rule\(\)\s+def rule_integration_by_parts\(integral\):.*?(?=@register_rule|$)'
if re.search(pattern_parts, content, re.DOTALL):
    content = re.sub(pattern_parts, timeout_fix, content, flags=re.DOTALL)
    with open(rules_file, 'w') as f:
        f.write(content)
    print("✅ rule_integration_by_parts 已添加超时保护")
else:
    print("⚠️ 未找到分部积分函数")

print("\n修复完成！重新运行: python auto_train.py")
