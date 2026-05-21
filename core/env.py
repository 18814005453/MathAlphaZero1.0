# core/env.py
from sympy import Integral, preorder_traversal, Symbol
from core.state import IntegrationState
from knowledge.rules import RULE_DICT, RULE_NAMES
from core.actions import Action


class IntegrationEnv:
    def __init__(self, max_steps=20):
        """
        初始化积分环境
        :param max_steps: 最大允许步数，防止无限循环
        """
        self.rules = RULE_DICT
        self.max_steps = max_steps
        self.steps_taken = 0
        self.original_expr = None  # 可选，用于计算全局奖励

    def reset(self, expr):
        """重置环境，准备新的积分问题"""
        self.steps_taken = 0
        self.original_expr = expr
        return IntegrationState(expr=expr)

    def is_terminal(self, state):
        """
        判断是否终止（积分完成或超出步数限制）
        :return: (is_terminal, reward)
        """
        if self.steps_taken >= self.max_steps:
            return True, -0.5  # 步数超限给予负奖励
        if not state.expr.has(Integral):
            return True, 1.0  # 成功消去所有积分，正奖励
        return False, 0.0

    def legal_actions(self, state):
        """
        利用 SymPy 匹配机制，返回所有可应用的动作。
        """
        expr = state.expr
        integrals = list(expr.atoms(Integral))
        actions = set()  # 使用 set 去重，防止同一规则对同一表达式多次添加

        for integral in integrals:
            for rule_name, rule_func in self.rules.items():
                result = rule_func(integral)
                if result is not None:
                    # ✅ 修复点 1：获取该规则的真实物理 ID，并遵循 Action(id, name) 的冻结数据类结构
                    rule_id = RULE_NAMES.index(rule_name)
                    action = Action(id=rule_id, name=rule_name)
                    actions.add(action)

        return list(actions)

    def step(self, state, action):
        """
        执行动作，返回 (next_state, reward, done, info)
        """
        self.steps_taken += 1
        expr = state.expr
        rule_name = action.name
        rule_func = self.rules[rule_name]

        current_integrals = list(expr.atoms(Integral))
        target_integral = None
        result = None

        # ✅ 修复点 2：在 step 中重新定位目标积分，完全解耦 Action 与目标状态
        for integral in current_integrals:
            res = rule_func(integral)
            if res is not None:
                target_integral = integral
                result = res
                break

        if target_integral is None or result is None:
            return state, -0.2, True, {"msg": "target_integral_not_found_or_invalid"}

        new_sub_expr, status = result

        # 替换原表达式中的积分部分
        if status == "substitution":
            # ✅ 修复点 3：解决换元状态下的符号污染与链式系数丢失问题
            # 将 u 临时转换为 x 交给网络继续处理，并乘上链式法则 factor
            u_sym = Symbol('u')
            temp_integral = new_sub_expr['integral'].subs(u_sym, Symbol('x'))
            new_expr = expr.subs(target_integral, new_sub_expr['factor'] * temp_integral)
        else:
            new_expr = expr.subs(target_integral, new_sub_expr)

        # 计算启发式奖励（基于节点数的变化）
        old_nodes = len(list(preorder_traversal(expr)))
        new_nodes = len(list(preorder_traversal(new_expr)))
        heuristic_reward = (old_nodes - new_nodes) / max(old_nodes, 1e-6)  # 节点减少比例

        # ✅ 修复点 4：继承并追加历史哈希，激活 MCTS 死循环防御机制
        new_hashes = set(state.history_hashes) if hasattr(state, 'history_hashes') and state.history_hashes else set()
        new_hashes.add(state.canonical_hash())

        next_state = IntegrationState(
            expr=new_expr,
            depth=state.depth + 1,
            history_hashes=new_hashes
        )

        # 判断是否终止
        done = not new_expr.has(Integral) or self.steps_taken >= self.max_steps
        if done and not new_expr.has(Integral):
            reward = 1.0 + heuristic_reward
        elif done and self.steps_taken >= self.max_steps:
            reward = -0.5 + heuristic_reward
        else:
            reward = heuristic_reward

        return next_state, reward, done, {"msg": "step_forward"}