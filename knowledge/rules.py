# knowledge/rules.py
import sympy
from sympy import Integral, Symbol, Add, Mul, Pow, sin, cos, tan, sec, csc, cot, coth
from sympy import log, exp, sqrt, atan, asin, acos, acot, asec, acsc
from sympy import sinh, cosh, tanh, sech, csch
from sympy import Abs, diff, apart, together, factor, expand, simplify, degree, Poly
from sympy.core.numbers import Rational, Number, NumberSymbol
from sympy.simplify.fu import TR2i, TR3, TR5, TR6, TR7, TR8, TR9, TR10, TR11

from knowledge.rule_registry import register_rule

x = Symbol('x')


# ---------- 辅助函数 ----------
def is_constant(expr, var):
    if isinstance(expr, (int, float)):
        return True
    if hasattr(expr, 'is_constant'):
        return expr.is_constant(var)
    return False


def is_linear_in_x(expr):
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
    if not expr.has(x):
        return (0, expr)
    if expr == x:
        return (1, 0)
    if expr.is_Mul and expr.args[0].is_constant() and expr.args[1] == x:
        return (expr.args[0], 0)
    a = diff(expr, x)
    if a.is_constant(x):
        b = expr.subs(x, 0)
        if simplify(expr - (a * x + b)) == 0:
            return (a, b)
    return (None, None)


def is_sqrt(expr):
    return isinstance(expr, Pow) and expr.exp == Rational(1, 2)


def is_rational_function(expr, var=x):
    num, den = expr.as_numer_denom()
    return num.is_polynomial(var) and den.is_polynomial(var)


# ---------- 规则函数 ----------
@register_rule()
def rule_extract_constant(integral):
    """∫ c f dx = c ∫ f dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    coeff, rest = func.as_independent(x, as_Add=False)
    if coeff != 1 and is_constant(coeff, x):
        return (coeff * Integral(rest, x), "rewrite")
    return None


@register_rule()
def rule_split_addition(integral):
    """∫ (f+g) dx = ∫f dx + ∫g dx"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Add):
        return (Add(*[Integral(arg, x) for arg in func.args]), "rewrite")
    return None


@register_rule()
def rule_linear_composition(integral):
    """∫ f(ax+b) dx → 换元"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    basic_funcs = (sin, cos, tan, sec, csc, cot, coth, exp, log,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, sech, csch)
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
                                "factor": 1 / a,
                                "integral": new_int
                            }, "substitution")
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
                            "factor": 1 / a,
                            "integral": new_int
                        }, "substitution")
    return None


@register_rule()
def rule_power_integral(integral):
    """幂函数积分 - 增强版"""
    if not isinstance(integral, Integral): return None
    func = integral.function
    if is_constant(func, x):
        return (func * x, "solved")
    if func == x or (isinstance(func, Pow) and func.base == x):
        if func == x:
            n = 1
        else:
            n = func.exp
        if n == -1:
            return (log(Abs(x)), "solved")
        else:
            return (x ** (n + 1) / (n + 1), "solved")
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if is_linear_in_x(base):
            a, b = linear_coeff(base)
            if a != 0 and is_constant(a, x):
                if expn == -1:
                    return ((1 / a) * log(Abs(base)), "solved")
                else:
                    return (base ** (expn + 1) / (a * (expn + 1)), "solved")
    if is_sqrt(func):
        inner = func.args[0]
        if is_linear_in_x(inner):
            a, b = linear_coeff(inner)
            if a != 0:
                return ((2 / (3 * a)) * (inner) ** (3 / 2), "solved")
    return None


@register_rule()
def rule_rational_power(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if expn.is_Rational and not expn.equals(-1):
            if base == x:
                return (x ** (expn + 1) / (expn + 1), "solved")
            if is_linear_in_x(base):
                a, b = linear_coeff(base)
                if a != 0:
                    return (base ** (expn + 1) / (a * (expn + 1)), "solved")
    return None


@register_rule()
def rule_trig_integral(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function

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

    if func.func == sin:
        res = linear_trig(sin, func.args[0])
        if res: return (res, "solved")
    if func.func == cos:
        res = linear_trig(cos, func.args[0])
        if res: return (res, "solved")
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
    if func == sec(x) ** 2:
        return (tan(x), "solved")
    if func == csc(x) ** 2:
        return (-cot(x), "solved")
    if func.is_Mul and len(func.args) == 2:
        if (func.args[0] == sec(x) and func.args[1] == tan(x)) or (func.args[0] == tan(x) and func.args[1] == sec(x)):
            return (sec(x), "solved")
        if (func.args[0] == csc(x) and func.args[1] == cot(x)) or (func.args[0] == cot(x) and func.args[1] == csc(x)):
            return (-csc(x), "solved")
    return None


@register_rule()
def rule_trig_power_reduction(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow) and func.exp == 2:
        base = func.base
        if base.func == sin:
            theta = base.args[0]
            return (Integral((1 - cos(2 * theta)) / 2, x), "rewrite")
        if base.func == cos:
            theta = base.args[0]
            return (Integral((1 + cos(2 * theta)) / 2, x), "rewrite")
        if base == tan(x):
            return (Integral(sec(x) ** 2 - 1, x), "rewrite")
        if base == cot(x):
            return (Integral(csc(x) ** 2 - 1, x), "rewrite")
    return None


@register_rule()
def rule_trig_product_to_sum(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    new_func = TR8(func)
    if new_func != func:
        return (Integral(new_func, x), "rewrite")
    return None


@register_rule()
def rule_exp_integral(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if func.func == exp:
        arg = func.args[0]
        if arg == x:
            return (exp(x), "solved")
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                return (exp(arg) / a, "solved")
    if isinstance(func, Pow):
        base, expn = func.base, func.exp
        if not base.has(x) and expn == x:
            return (func / log(base), "solved")
    return None


@register_rule()
def rule_log_integral(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if func.func == log:
        arg = func.args[0]
        if arg == x:
            return (x * log(x) - x, "solved")
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                return ((arg / a) * log(arg) - arg / a, "solved")
    return None


@register_rule()
def rule_exp_trig_product(integral):
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
            theta = trig_part.args[0]
            if not is_linear_in_x(theta):
                return None
            beta = diff(theta, x)
            if not is_constant(beta, x):
                return None
            gamma = theta.subs(x, 0)
            if trig_part.func == sin:
                result = exp_part * (alpha * sin(theta) - beta * cos(theta)) / (alpha ** 2 + beta ** 2)
                return (result, "solved")
            if trig_part.func == cos:
                result = exp_part * (alpha * cos(theta) + beta * sin(theta)) / (alpha ** 2 + beta ** 2)
                return (result, "solved")
    return None


from sympy import Wild, log, sqrt, Abs, asin, atan, Rational, Pow, Add


@register_rule()
def rule_inv_trig_integral(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    x = integral.limits[0][0]
    A = Wild('A', exclude=[x])
    if isinstance(func, Pow) and func.exp == -Rational(1, 2):
        inner = func.base
        m1 = inner.match(A - x ** 2)
        if m1 and m1[A] != 0:
            a = sqrt(m1[A])
            return (asin(x / a), "solved")
        m2 = inner.match(x ** 2 + A)
        if m2 and m2[A] != 0:
            if '-' not in str(m2[A]):
                return (log(x + sqrt(x ** 2 + m2[A])), "solved")
            else:
                return (log(Abs(x + sqrt(x ** 2 + m2[A]))), "solved")
        m3 = inner.match(x ** 2 - A)
        if m3 and m3[A] != 0:
            return (log(Abs(x + sqrt(x ** 2 - m3[A]))), "solved")
    if isinstance(func, Pow) and func.exp == -1:
        denom = func.base
        m4 = denom.match(x ** 2 + A)
        if m4 and m4[A] != 0:
            a = sqrt(m4[A])
            return ((1 / a) * atan(x / a), "solved")
    return None


@register_rule()
def rule_rational_function(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if is_rational_function(func, x):
        try:
            decomposed = apart(func, x)
            if decomposed != func and isinstance(decomposed, Add):
                return (Add(*[Integral(term, x) for term in decomposed.args]), "rewrite")
            if isinstance(func, Pow) and func.exp == -1 and is_linear_in_x(func.base):
                a, b = linear_coeff(func.base)
                if a != 0:
                    return ((1 / a) * log(Abs(func.base)), "solved")
        except:
            pass
    return None


@register_rule()
def rule_rational_improper(integral):
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


@register_rule()
def rule_sqrt_quadratic(integral):
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
            a = poly_expr.coeff_monomial(x ** 2)
            b = poly_expr.coeff_monomial(x)
            c = poly_expr.coeff_monomial(1)
            h = -b / (2 * a)
            k = c - b ** 2 / (4 * a)
            u_sym = Symbol('u')
            new_inner = a * u_sym ** 2 + k
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
    dv_dx = Mul(*(args[:u_idx] + args[u_idx + 1:]))

    try:
        v = sympy.integrate(dv_dx, x)
        if v.has(Integral):
            return None
    except:
        return None

    du_dx = diff(u, x)
    new_expr = u * v - Integral(v * du_dx, x)
    return (new_expr, "rewrite")


@register_rule()
def rule_trig_substitution(integral):
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
        if inner.equals(1 - x ** 2):
            theta = Symbol('theta')
            x_theta = sin(theta)
            dx_dtheta = cos(theta)
            new_integrand = func.subs(x, x_theta) * dx_dtheta
            new_integrand = new_integrand.replace(sqrt(cos(theta) ** 2), cos(theta))
            return ({
                        "type": "substitution",
                        "u_expr": asin(x),
                        "factor": 1,
                        "integral": Integral(new_integrand, theta)
                    }, "substitution")
    return None


@register_rule()
def rule_hyperbolic_integral(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function

    def linear_hyper(fn, arg):
        if arg == x:
            if fn == sinh: return cosh(x)
            if fn == cosh: return sinh(x)
            if fn == tanh: return log(cosh(x))
            if fn == coth: return log(sinh(x))
            if fn == sech: return atan(sinh(x))
            if fn == csch: return log(tanh(x / 2))
        if is_linear_in_x(arg):
            a, b = linear_coeff(arg)
            if a != 0:
                if fn == sinh: return cosh(arg) / a
                if fn == cosh: return sinh(arg) / a
                if fn == tanh: return log(cosh(arg)) / a
                if fn == coth: return log(sinh(arg)) / a
                if fn == sech: return atan(sinh(arg)) / a
                if fn == csch: return log(tanh(arg / 2)) / a
        return None

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
    if func == sech(x) ** 2:
        return (tanh(x), "solved")
    if func == csch(x) ** 2:
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


@register_rule()
def rule_reduction_formula(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    if isinstance(func, Pow) and func.exp.is_number and func.exp > 1:
        base, n = func.base, func.exp
        if base.func == sin:
            theta = base.args[0]
            if not is_linear_in_x(theta):
                return None
            a, _ = linear_coeff(theta)
            if a == 0:
                return None
            term1 = - (1 / n) * sin(theta) ** (n - 1) * cos(theta) / a
            term2 = ((n - 1) / n) * Integral(sin(theta) ** (n - 2), x)
            return (term1 + term2, "rewrite")
        if base.func == cos:
            theta = base.args[0]
            if not is_linear_in_x(theta):
                return None
            a, _ = linear_coeff(theta)
            if a == 0:
                return None
            term1 = (1 / n) * cos(theta) ** (n - 1) * sin(theta) / a
            term2 = ((n - 1) / n) * Integral(cos(theta) ** (n - 2), x)
            return (term1 + term2, "rewrite")
        if base == tan(x):
            if n != 1:
                term1 = (1 / (n - 1)) * tan(x) ** (n - 1)
                term2 = Integral(tan(x) ** (n - 2), x)
                return (term1 - term2, "rewrite")
    return None


@register_rule()
def rule_simplify(integral):
    if not isinstance(integral, Integral): return None
    func = integral.function
    simp = simplify(func)
    if simp != func:
        return (Integral(simp, x), "rewrite")
    trig_simp = TR2i(func)
    if trig_simp != func:
        return (Integral(trig_simp, x), "rewrite")
    return None


# ========== 构建动作空间（必须在所有规则定义之后） ==========
from knowledge.rule_registry import build_action_space

build_action_space()