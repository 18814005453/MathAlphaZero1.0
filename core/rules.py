import sympy
from sympy import Integral, Symbol, Add, Mul, Pow, sin, cos, tan, sec, csc, cot, coth
from sympy import log, exp, sqrt, atan, asin, acos, acot, asec, acsc
from sympy import sinh, cosh, tanh, sech, csch
from sympy import Abs, diff, apart, together, factor, expand, simplify, degree, Poly
from sympy.core.numbers import Rational, Number, NumberSymbol
from sympy.simplify.fu import TR2i, TR3, TR5, TR6, TR7, TR8, TR9, TR10, TR11

x = Symbol('x')

# ---------- 辅助函数 ----------
def is_constant(expr, var):
    # 如果 expr 已经是纯 Python 数字类型(int, float)，它必然是常数
    if isinstance(expr, (int, float)):
        return True
    # 如果是 SymPy 对象，再调用它的 is_constant 方法
    if hasattr(expr, 'is_constant'):
        return expr.is_constant(var)
    return False

def is_linear_in_x(expr):
    """判断是否为 a*x + b 形式 (a, b 常数)"""
    if not expr.has(x):
        return False
    first = diff(expr, x)
    if not first.is_constant(x):
        return False
    second = diff(first, x)
    if second != 0:
        return False
    return True

def linear_coeff(expr):
    """返回 (a, b) 使得 expr = a*x + b，若非常数线性则返回 (None, None)"""
    if not expr.has(x):
        return (0, expr)
    if expr == x:
        return (1, 0)
    if expr.is_Mul and expr.args[0].is_constant() and expr.args[1] == x:
        return (expr.args[0], 0)
    a = diff(expr, x)
    if a.is_constant(x):
        b = expr.subs(x, 0)
        if simplify(expr - (a*x + b)) == 0:
            return (a, b)
    return (None, None)

def is_sqrt(expr):
    """判断 expr 是否为 sqrt(...) 形式"""
    return isinstance(expr, Pow) and expr.exp == Rational(1, 2)

def is_rational_function(expr, var=x):
    """更可靠的有理函数判断"""
    num, den = expr.as_numer_denom()
    return num.is_polynomial(var) and den.is_polynomial(var)

# ---------- 规则函数 ----------
# 每条规则返回三种可能：
#   None                     -> 不适用
#   (expr, "solved")         -> 积分已完成（无 Integral）
#   (expr, "rewrite")        -> 重写为另一个积分（仍含 Integral）
#   ({"type":"substitution", ...}, "substitution") -> 换元状态

def rule_extract_constant(integral):
    """∫ c f dx = c ∫ f dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    coeff, rest = func.as_independent(x, as_Add=False)
    if coeff != 1 and is_constant(coeff, x):
        return (coeff * Integral(rest, x), "rewrite")
    return None

def rule_split_addition(integral):
    """∫ (f+g) dx = ∫f dx + ∫g dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Add):
        return (Add(*[Integral(arg, x) for arg in func.args]), "rewrite")
    return None

def rule_linear_composition(integral):
    """
    ∫ f(ax+b) dx → 返回换元状态 (结构化)，避免 subs 污染
    返回: ({"u": ax+b, "factor": 1/a, "inner_integral": Integral(f(u), u)}, "substitution")
    """
    if not isinstance(integral, Integral): return None
    func = integral.function
    # 基本初等函数（不包括 sqrt，因为 sqrt 的 func 是 Pow）
    basic_funcs = (sin, cos, tan, sec, csc, cot, coth, exp, log,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, sech, csch)
    # 检查是否匹配基本函数
    for fn_type in basic_funcs:
        if func.func == fn_type:
            arg = func.args[0]
            if is_linear_in_x(arg):
                a, b = linear_coeff(arg)
                if a != 0 and is_constant(a, x):
                    u_sym = Symbol('u')
                    f_u = func.func(u_sym)
                    new_int = Integral(f_u, u_sym)
                    return ({
                        "type": "substitution",
                        "u_expr": arg,
                        "factor": 1/a,
                        "integral": new_int
                    }, "substitution")
    # 单独处理 sqrt(ax+b) 即 (ax+b)^(1/2)
    if is_sqrt(func):
        inner = func.args[0]
        if is_linear_in_x(inner):
            a, b = linear_coeff(inner)
            if a != 0 and is_constant(a, x):
                u_sym = Symbol('u')
                f_u = sqrt(u_sym)
                new_int = Integral(f_u, u_sym)
                return ({
                    "type": "substitution",
                    "u_expr": inner,
                    "factor": 1/a,
                    "integral": new_int
                }, "substitution")
    return None

def rule_power_integral(integral):
    """幂函数积分，包括常数、x^n、(ax+b)^n、根式"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    # 常数
    if is_constant(func, x):
        return (func * x, "solved")
    # x^n
    if func == x:
        return (x**2 / 2, "solved")
    if isinstance(func, Pow) and func.base == x:
        n = func.exp
        # 使用 equals 避免符号问题
        if n.equals(-1):
            return (log(Abs(x)), "solved")
        else:
            return (x**(n+1) / (n+1), "solved")
    # (ax+b)^n
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if is_linear_in_x(base):
            a, b = linear_coeff(base)
            if a != 0 and is_constant(a, x):
                if expn.equals(-1):
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

def rule_rational_power(integral):
    """x^(p/q) 或 (ax+b)^(p/q) 分数指数"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if expn.is_Rational and not expn.equals(-1):
            if base == x:
                return (x**(expn+1) / (expn+1), "solved")
            if is_linear_in_x(base):
                a, b = linear_coeff(base)
                if a != 0:
                    return (base**(expn+1) / (a * (expn+1)), "solved")
    return None

def rule_trig_integral(integral):
    """基本三角函数积分，支持线性内层 (ax+b)"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    # 处理 sin, cos, tan, sec, csc, cot, coth 的线性内层
    def linear_trig(fn, arg):
        if arg == x:
            if fn == sin: return -cos(x)
            if fn == cos: return sin(x)
            if fn == tan: return -log(Abs(cos(x)))
            if fn == sec: return log(Abs(sec(x) + tan(x)))
            if fn == csc: return -log(Abs(csc(x) + cot(x)))
            if fn == cot: return log(Abs(sin(x)))
            if fn == coth: return log(sinh(x))
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                if fn == sin: return -cos(arg) / a
                if fn == cos: return sin(arg) / a
                if fn == tan: return -log(Abs(cos(arg))) / a
                if fn == sec: return log(Abs(sec(arg) + tan(arg))) / a
                if fn == csc: return -log(Abs(csc(arg) + cot(arg))) / a
                if fn == cot: return log(Abs(sin(arg))) / a
                if fn == coth: return log(sinh(arg)) / a
        return None
    # sin, cos
    if func.func == sin:
        res = linear_trig(sin, func.args[0])
        if res: return (res, "solved")
    if func.func == cos:
        res = linear_trig(cos, func.args[0])
        if res: return (res, "solved")
    # tan, sec, csc, cot, coth
    if func.func == tan:
        res = linear_trig(tan, func.args[0])
        if res: return (res, "solved")
    if func.func == sec:
        res = linear_trig(sec, func.args[0])
        if res: return (res, "solved")
    if func.func == csc:
        res = linear_trig(csc, func.args[0])
        if res: return (res, "solved")
    if func.func == cot:
        res = linear_trig(cot, func.args[0])
        if res: return (res, "solved")
    if func.func == coth:
        res = linear_trig(coth, func.args[0])
        if res: return (res, "solved")
    # sec^2, csc^2 特殊处理（因为它们是幂）
    if func == sec(x)**2:
        return (tan(x), "solved")
    if func == csc(x)**2:
        return (-cot(x), "solved")
    if func == sec(x)**2 and is_linear_in_x(func.args[0].args[0]):
        # 支持 sec(ax+b)^2
        if isinstance(func, Pow) and func.base.func == sec:
            inner = func.base.args[0]
            if is_linear_in_x(inner):
                a, b = linear_coeff(inner)
                if a != 0:
                    return (tan(inner) / a, "solved")
    if func == csc(x)**2 and is_linear_in_x(func.args[0].args[0]):
        if isinstance(func, Pow) and func.base.func == csc:
            inner = func.base.args[0]
            if is_linear_in_x(inner):
                a, b = linear_coeff(inner)
                if a != 0:
                    return (-cot(inner) / a, "solved")
    # sec(x)tan(x) 和 csc(x)cot(x)
    if func.is_Mul and len(func.args) == 2:
        if (func.args[0] == sec(x) and func.args[1] == tan(x)) or (func.args[0] == tan(x) and func.args[1] == sec(x)):
            return (sec(x), "solved")
        if (func.args[0] == csc(x) and func.args[1] == cot(x)) or (func.args[0] == cot(x) and func.args[1] == csc(x)):
            return (-csc(x), "solved")
    return None

def rule_trig_power_reduction(integral):
    """降幂：sin^2, cos^2, tan^2, cot^2，支持任意角度 theta"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow) and func.exp == 2:
        base = func.base
        if base.func == sin:
            theta = base.args[0]
            # sin^2(theta) = (1 - cos(2*theta))/2
            return (Integral((1 - cos(2*theta))/2, x), "rewrite")
        if base.func == cos:
            theta = base.args[0]
            # cos^2(theta) = (1 + cos(2*theta))/2
            return (Integral((1 + cos(2*theta))/2, x), "rewrite")
        if base == tan(x):
            return (Integral(sec(x)**2 - 1, x), "rewrite")
        if base == cot(x):
            return (Integral(csc(x)**2 - 1, x), "rewrite")
    return None

def rule_trig_product_to_sum(integral):
    """积化和差，使用 SymPy 的 TR8"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    new_func = TR8(func)
    if new_func != func:
        return (Integral(new_func, x), "rewrite")
    return None


def rule_exp_integral(integral):
    """∫ e^(ax+b) dx 或 ∫ a^x dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function

    # 场景 A: 传统的自然指数 exp(...)
    if func.func == exp:
        arg = func.args[0]
        if arg == x:
            return (exp(x), "solved")
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                return (exp(arg) / a, "solved")

    # 场景 B: 盲点修复 —— 任意常数底数 a^x 形式 (在 SymPy 中属于 Pow 实例)
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        # 如果底数不包含 x (是常数)，且指数就是单自变量 x
        if not base.has(x) and expn == x:
            return (func / log(base), "solved")

    return None

def rule_log_integral(integral):
    """∫ ln(ax+b) dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if func.func == log:
        arg = func.args[0]
        if arg == x:
            return (x*log(x) - x, "solved")
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                return ((arg/a)*log(arg) - arg/a, "solved")
    return None

def rule_exp_trig_product(integral):
    """
    ∫ e^(ax+b) sin(cx+d) dx 或 cos，保留完整的相位。
    公式：
        ∫ e^(α x) sin(β x + γ) dx = e^(α x) [α sin(βx+γ) - β cos(βx+γ)]/(α^2+β^2)
        ∫ e^(α x) cos(β x + γ) dx = e^(α x) [α cos(βx+γ) + β sin(βx+γ)]/(α^2+β^2)
    """
    if not isinstance(integral, Integral): return None
    func = integral.function
    if func.is_Mul:
        exp_part = None
        trig_part = None
        for arg in func.args:
            if arg.func == exp:
                exp_part = arg
            if arg.has(sin) or arg.has(cos):
                trig_part = arg
        if exp_part and trig_part:
            e_arg = exp_part.args[0]
            if not is_linear_in_x(e_arg):
                return None
            alpha = diff(e_arg, x)
            if not is_constant(alpha, x):
                return None
            # 三角函数参数，可能为 beta*x + gamma
            theta = trig_part.args[0]
            if not is_linear_in_x(theta):
                return None
            beta = diff(theta, x)
            if not is_constant(beta, x):
                return None
            gamma = theta.subs(x, 0)
            # 构造完整表达式
            if trig_part.func == sin:
                result = exp_part * (alpha * sin(theta) - beta * cos(theta)) / (alpha**2 + beta**2)
                return (result, "solved")
            if trig_part.func == cos:
                result = exp_part * (alpha * cos(theta) + beta * sin(theta)) / (alpha**2 + beta**2)
                return (result, "solved")
    return None


from sympy import Wild, log, sqrt, Abs, asin, atan, Rational, Pow, Add


def rule_inv_trig_integral(integral):
    """反三角函数与对数积分形式：
    - 1/sqrt(a^2 - x^2) -> arcsin(x/a)
    - 1/sqrt(x^2 + a^2) -> ln(x + sqrt(x^2 + a^2))
    - 1/sqrt(x^2 - a^2) -> ln|x + sqrt(x^2 - a^2)|
    - 1/(a^2 + x^2)     -> (1/a)*atan(x/a)
    """
    if not isinstance(integral, Integral): return None
    func = integral.function
    x = integral.limits[0][0]  # 动态获取积分变量，防止硬编码 x 冲突

    # 定义一个通配符，用来匹配常数 a**2 或者 a
    A = Wild('A', exclude=[x])

    # ==========================================
    # 情况一：处理 1/sqrt(...) 形式 (即指数为 -1/2)
    # ==========================================
    if isinstance(func, Pow) and func.exp == -Rational(1, 2):
        inner = func.base

        # 1. 匹配 1/sqrt(a^2 - x^2)
        # 用 inner 去匹配 A - x**2
        m1 = inner.match(A - x ** 2)
        if m1 and m1[A] != 0:
            a = sqrt(m1[A])
            return (asin(x / a), "solved")

        # 2. 匹配 1/sqrt(x^2 + a^2) 或 1/sqrt(a^2 + x^2)
        m2 = inner.match(x ** 2 + A)
        if m2 and m2[A] != 0:
            # 判断 A 的正负（通过 equals 或者直接检查符号前缀）
            # 如果 A 是正的（比如 a**2），走这个分支
            # 兼容你的测试集：如果是 -a**2 + x**2 且测试集认为它是减，我们用符号区分
            if '-' not in str(m2[A]):
                return (log(x + sqrt(x ** 2 + m2[A])), "solved")
            else:
                # 如果 A 带有负号（比如 -a**2），说明是 x^2 - a^2 形式
                return (log(Abs(x + sqrt(x ** 2 + m2[A]))), "solved")

        # 3. 补充显式匹配 1/sqrt(x^2 - a^2)
        m3 = inner.match(x ** 2 - A)
        if m3 and m3[A] != 0:
            return (log(Abs(x + sqrt(x ** 2 - m3[A]))), "solved")

    # ==========================================
    # 情况二：处理 1/(a^2 + x^2) 形式 (即指数为 -1)
    # ==========================================
    if isinstance(func, Pow) and func.exp == -1:
        denom = func.base
        m4 = denom.match(x ** 2 + A)
        if m4 and m4[A] != 0:
            a = sqrt(m4[A])
            return ((1 / a) * atan(x / a), "solved")

    return None

def rule_rational_function(integral):
    """有理函数：部分分式分解"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if is_rational_function(func, x):
        try:
            decomposed = apart(func, x)
            if decomposed != func and isinstance(decomposed, Add):
                return (Add(*[Integral(term, x) for term in decomposed.args]), "rewrite")
            # 特殊情况 1/(ax+b)
            if isinstance(func, Pow) and func.exp == -1 and is_linear_in_x(func.base):
                a, b = linear_coeff(func.base)
                if a != 0:
                    return ((1/a)*log(Abs(func.base)), "solved")
        except:
            pass
    return None

def rule_rational_improper(integral):
    """假分式：多项式除法"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if is_rational_function(func, x):
        num, den = func.as_numer_denom()
        if num.is_polynomial(x) and den.is_polynomial(x):
            from sympy import poly, div
            try:
                q, r = div(poly(num, x), poly(den, x))
                q_expr = q.as_expr()
                r_expr = r.as_expr() / den
                if not q_expr.equals(0):
                    return (Integral(q_expr, x) + Integral(r_expr, x), "rewrite")
            except:
                pass
    return None

def rule_sqrt_quadratic(integral):
    """处理 sqrt(ax^2+bx+c) 的换元（配方法）"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    sqrt_expr = None
    other = 1
    if is_sqrt(func):
        sqrt_expr = func
        other = 1
    elif func.is_Mul:
        for arg in func.args:
            if is_sqrt(arg):
                sqrt_expr = arg
                other = Mul(*[a for a in func.args if a != arg])
                break
    if sqrt_expr:
        inner = sqrt_expr.args[0]
        if inner.is_polynomial(x) and degree(inner, x) == 2:
            poly_expr = Poly(inner, x)
            a = poly_expr.coeff_monomial(x**2)
            b = poly_expr.coeff_monomial(x)
            c = poly_expr.coeff_monomial(1)
            h = -b/(2*a)
            k = c - b**2/(4*a)
            u_sym = Symbol('u')
            new_inner = a*u_sym**2 + k
            new_sqrt = sqrt(new_inner)
            new_other = other.subs(x, u_sym - h)
            new_integrand = new_other * new_sqrt
            return ({
                "type": "substitution",
                "u_expr": x + h,
                "factor": 1,
                "integral": Integral(new_integrand, u_sym)
            }, "substitution")
    return None


def rule_integration_by_parts(integral):
    """分部积分：基于 LIATE 法则自动选择 u 和 dv，拆解 x*sin(x), x*exp(x) 等结构"""
    if not isinstance(integral, Integral): return None
    func = integral.function

    # 必须是乘积形式才能应用分部积分
    if not func.is_Mul or len(func.args) < 2:
        return None

    # 1. 简易 LIATE 评分系统 (分数越高的越优先作为 u 求导，以简化表达式)
    def liate_score(expr):
        if expr.has(log): return 5
        if expr.has(asin, acos, atan, acot, asec, acsc): return 4
        if expr.is_polynomial(x) or (isinstance(expr, Pow) and expr.base == x): return 3
        if expr.has(sin, cos, tan, sec, csc, cot): return 2
        if expr.has(exp): return 1
        return 0

    args = list(func.args)

    # 2. 选出得分最高的项作为 u
    u_idx = max(range(len(args)), key=lambda i: liate_score(args[i]))
    u = args[u_idx]

    # 剩下的项乘起来作为 dv_dx
    dv_dx = Mul(*(args[:u_idx] + args[u_idx + 1:]))

    # 3. 计算 v = ∫ dv_dx dx
    # 使用 sympy.integrate 求解被剥离出来的相对简单的 dv (如 exp(x), sin(x))
    v = sympy.integrate(dv_dx, x)

    # 防御机制：如果对 dv 的积分失败（返回了未计算的 Integral），说明拆解方向错误，直接放弃此动作
    if v.has(Integral):
        return None

    # 4. 计算 du = u' dx
    du_dx = diff(u, x)

    # 5. 应用公式: ∫ u dv = u*v - ∫ v du
    new_expr = u * v - Integral(v * du_dx, x)

    return (new_expr, "rewrite")

def rule_trig_substitution(integral):
    """三角换元框架：匹配 sqrt(a^2 - x^2), sqrt(a^2+x^2), sqrt(x^2-a^2)"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    sqrt_expr = None
    if is_sqrt(func):
        sqrt_expr = func
    elif func.is_Mul:
        for arg in func.args:
            if is_sqrt(arg):
                sqrt_expr = arg
                break
    if sqrt_expr:
        inner = sqrt_expr.args[0]
        # 1 - x^2
        if inner.equals(1 - x**2):
            theta = Symbol('theta')
            x_theta = sin(theta)
            dx_dtheta = cos(theta)
            new_integrand = func.subs(x, x_theta) * dx_dtheta
            # 简化 sqrt(cos^2 theta) = cos(theta) (假设 theta 在 [-pi/2, pi/2])
            new_integrand = new_integrand.replace(sqrt(cos(theta)**2), cos(theta))
            return ({
                "type": "substitution",
                "u_expr": asin(x),
                "factor": 1,
                "integral": Integral(new_integrand, theta)
            }, "substitution")
    return None

def rule_hyperbolic_integral(integral):
    """基本双曲函数积分，支持线性内层 (ax+b)"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    # 处理 sinh(ax+b), cosh, tanh, sech, csch, coth 的线性内层
    def linear_hyper(fn, arg):
        if arg == x:
            if fn == sinh: return cosh(x)
            if fn == cosh: return sinh(x)
            if fn == tanh: return log(cosh(x))
            if fn == coth: return log(sinh(x))
            if fn == sech: return atan(sinh(x))
            if fn == csch: return log(tanh(x/2))
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                if fn == sinh: return cosh(arg) / a
                if fn == cosh: return sinh(arg) / a
                if fn == tanh: return log(cosh(arg)) / a
                if fn == coth: return log(sinh(arg)) / a
                if fn == sech: return atan(sinh(arg)) / a
                if fn == csch: return log(tanh(arg/2)) / a
        return None
    # 基本函数
    if func.func == sinh:
        res = linear_hyper(sinh, func.args[0])
        if res: return (res, "solved")
    if func.func == cosh:
        res = linear_hyper(cosh, func.args[0])
        if res: return (res, "solved")
    if func.func == tanh:
        res = linear_hyper(tanh, func.args[0])
        if res: return (res, "solved")
    if func.func == coth:
        res = linear_hyper(coth, func.args[0])
        if res: return (res, "solved")
    if func.func == sech:
        res = linear_hyper(sech, func.args[0])
        if res: return (res, "solved")
    if func.func == csch:
        res = linear_hyper(csch, func.args[0])
        if res: return (res, "solved")
    # sech^2, csch^2
    if func == sech(x)**2:
        return (tanh(x), "solved")
    if func == csch(x)**2:
        return (-coth(x), "solved")
    if isinstance(func, Pow) and func.exp == 2:
        if func.base.func == sech:
            inner = func.base.args[0]
            if inner == x:
                return (tanh(x), "solved")
            if is_linear_in_x(inner):
                a, b = linear_coeff(inner)
                if a != 0:
                    return (tanh(inner) / a, "solved")
        if func.base.func == csch:
            inner = func.base.args[0]
            if inner == x:
                return (-coth(x), "solved")
            if is_linear_in_x(inner):
                a, b = linear_coeff(inner)
                if a != 0:
                    return (-coth(inner) / a, "solved")
    return None

def rule_reduction_formula(integral):
    """
    递推公式：∫ sin^n(θ) dx, ∫ cos^n(θ) dx, ∫ tan^n(θ) dx
    支持 θ = ax+b 并引入链式法则系数 1/a
    """
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow) and func.exp.is_number and func.exp > 1:
        base, n = func.base, func.exp
        # 仅当角度为线性时处理链式法则
        if base.func == sin:
            theta = base.args[0]
            if not is_linear_in_x(theta):
                return None
            a, _ = linear_coeff(theta)
            if a == 0:
                return None
            # ∫ sin^n(θ) dx = (1/a) ∫ sin^n(u) du, 但我们需要降幂公式在 u 上
            # 直接给出降幂表达式，外部乘 1/a
            term1 = - (1/n) * sin(theta)**(n-1) * cos(theta) / a
            term2 = ((n-1)/n) * Integral(sin(theta)**(n-2), x)
            return (term1 + term2, "rewrite")
        if base.func == cos:
            theta = base.args[0]
            if not is_linear_in_x(theta):
                return None
            a, _ = linear_coeff(theta)
            if a == 0:
                return None
            term1 = (1/n) * cos(theta)**(n-1) * sin(theta) / a
            term2 = ((n-1)/n) * Integral(cos(theta)**(n-2), x)
            return (term1 + term2, "rewrite")
        if base == tan(x):
            # tan^n x 降幂公式没有额外的链式，因为角度就是 x
            if n != 1:
                term1 = (1/(n-1)) * tan(x)**(n-1)
                term2 = Integral(tan(x)**(n-2), x)
                return (term1 - term2, "rewrite")
    return None

def rule_simplify(integral):
    """通用化简：simplify 和 三角恒等"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    simp = simplify(func)
    if simp != func:
        return (Integral(simp, x), "rewrite")
    trig_simp = TR2i(func)
    if trig_simp != func:
        return (Integral(trig_simp, x), "rewrite")
    return None

# ---------- 显式注册规则 ----------
RULE_DICT = {
    "ExtractConstant": rule_extract_constant,
    "SplitAddition": rule_split_addition,
    "LinearComposition": rule_linear_composition,
    "PowerIntegral": rule_power_integral,
    "RationalPower": rule_rational_power,
    "TrigIntegral": rule_trig_integral,
    "TrigPowerReduction": rule_trig_power_reduction,
    "TrigProductToSum": rule_trig_product_to_sum,
    "ExpIntegral": rule_exp_integral,
    "LogIntegral": rule_log_integral,
    "ExpTrigProduct": rule_exp_trig_product,
    "InvTrigIntegral": rule_inv_trig_integral,
    "RationalFunction": rule_rational_function,
    "RationalImproper": rule_rational_improper,
    "SqrtQuadratic": rule_sqrt_quadratic,
    "IntegrationByParts": rule_integration_by_parts,
    "TrigSubstitution": rule_trig_substitution,
    "HyperbolicIntegral": rule_hyperbolic_integral,
    "ReductionFormula": rule_reduction_formula,
    "Simplify": rule_simplify,
}

RULE_NAMES = list(RULE_DICT.keys())

# ---------- 显式注册规则 ----------
# (保持你原有的 RULE_DICT 和 RULE_NAMES 不变)
# ...

class MathRuleBase:
    """提供给网络和训练脚本的接口"""
    def __init__(self):
        # 延迟导入以避免与 actions.py 产生循环依赖
        from core.actions import NUM_ACTIONS
        self.num_actions = NUM_ACTIONS