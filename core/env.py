from core.rules import RULE_DICT
from core.actions import ACTIONS
from sympy import Integral


class IntegrationEnv:
    def __init__(self):
        self.rules = RULE_DICT

    def is_terminal(self, state):
        if not hasattr(state.expr, 'has'):
            return True, 1.0
        # 只要表达式里彻底没有积分号了，就大功告成
        if not state.expr.has(Integral):
            return True, 1.0
        return False, 0.0

    def legal_actions(self, state):
        return ACTIONS

    def step(self, state, action):
        expr = state.expr
        rule_name = action.name
        rule_func = self.rules[rule_name]

        # 1. 核心修复：自动寻找表达式中还没解开的积分项
        integrals = expr.atoms(Integral)
        if not integrals:
            return state, 0.0, True, {"msg": "already_solved"}

        # 2. 锁定需要处理的那个积分（默认先处理第一个）
        target_integral = list(integrals)[0]

        # 3. 把纯粹的 Integral 扔给规则去算
        result = rule_func(target_integral)

        if result is None:
            return state, -0.1, False, {"msg": "rule_not_applicable"}

        new_sub_expr, status = result

        # 4. 完美缝合：将算完的新部分替换回原来的大式子里
        if status == "substitution":
            new_full_expr = expr.subs(target_integral, new_sub_expr['integral'])
        else:
            new_full_expr = expr.subs(target_integral, new_sub_expr)

        from core.state import IntegrationState
        next_state = IntegrationState(expr=new_full_expr)

        # 5. 判断这步走完，是不是整道题都做完了
        if not new_full_expr.has(Integral):
            return next_state, 1.0, True, {"msg": "fully_solved"}
        else:
            return next_state, 0.0, False, {"msg": "step_forward"}

    def reset(self, expr):
        return type('obj', (object,), {'expr': expr})()