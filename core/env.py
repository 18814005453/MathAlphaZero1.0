# core/env.py
import math
import time
from sympy import Integral, Symbol
from core.state import IntegrationState
from core.actions import Action
from knowledge.rule_registry import get_rule_by_name, get_all_rule_names
from typing import Tuple, Optional, Dict, Any, List

class IntegrationEnv:
    """
    符号积分环境，支持改进的奖励塑造和终止条件。
    """
    def __init__(
        self,
        max_steps: int = 20,
        time_limit: Optional[float] = None,
        simplify_reward_alpha: float = 0.3,      # 化简奖励系数
        simplify_reward_beta: float = 2.0,       # 化简奖励指数衰减系数
        step_penalty: float = 0.02,              # 每步惩罚
        final_bonus: float = 1.0,                # 成功完成奖励
        invalid_action_penalty: float = 0.2,     # 无效动作惩罚
        partial_completion_bonus: float = 0.2    # 部分化简奖励（超时/步数超限时）
    ):
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.simplify_alpha = simplify_reward_alpha
        self.simplify_beta = simplify_reward_beta
        self.step_penalty = step_penalty
        self.final_bonus = final_bonus
        self.invalid_action_penalty = invalid_action_penalty
        self.partial_completion_bonus = partial_completion_bonus

        self.steps_taken = 0
        self.start_time = None
        self.initial_complexity = None

        # 规则名称到ID的映射（用于快速生成合法动作）
        self._rule_name_to_id = {name: idx for idx, name in enumerate(get_all_rule_names())}

    def reset(self, expr) -> IntegrationState:
        """重置环境，返回初始状态"""
        self.steps_taken = 0
        self.start_time = time.time() if self.time_limit else None
        state = IntegrationState(expr=expr, depth=0, history_hashes=set())
        self.initial_complexity = state.ast_complexity()
        return state

    def _compute_simplify_reward(self, old_complexity: int, new_complexity: int) -> float:
        """
        计算化简奖励：指数型收益，避免线性饱和。
        当复杂度降低时为正奖励，增加时为负奖励（较小惩罚）。
        """
        if new_complexity < old_complexity:
            reduction_ratio = (old_complexity - new_complexity) / max(old_complexity, 1)
            # 奖励随化简比例指数增长但上限为 0.5
            r = self.simplify_alpha * (1 - math.exp(-self.simplify_beta * reduction_ratio))
            return min(r, 0.5)
        elif new_complexity > old_complexity:
            increase_ratio = (new_complexity - old_complexity) / max(old_complexity, 1)
            return -0.1 * min(increase_ratio, 1.0)
        else:
            return 0.0

    def is_terminal(self, state: IntegrationState) -> Tuple[bool, float]:
        """
        判断是否终止，并返回终止奖励。
        终止条件：
        - 步数超限 -> 终止，给予部分化简奖励（如果化简有效）
        - 时间超限 -> 终止，给予部分化简奖励
        - 表达式中不含积分号 -> 成功完成，给予最终奖励 + 化简比例奖励
        """
        # 步数超限
        if self.steps_taken >= self.max_steps:
            if self.initial_complexity is not None:
                current_c = state.ast_complexity()
                reduction = (self.initial_complexity - current_c) / max(self.initial_complexity, 1)
                partial_reward = self.partial_completion_bonus * max(0, reduction)
            else:
                partial_reward = 0.0
            return True, partial_reward - 0.1   # 步数超限小惩罚

        # 时间超限
        if self.time_limit is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                if self.initial_complexity is not None:
                    current_c = state.ast_complexity()
                    reduction = (self.initial_complexity - current_c) / max(self.initial_complexity, 1)
                    partial_reward = self.partial_completion_bonus * max(0, reduction)
                else:
                    partial_reward = 0.0
                return True, partial_reward - 0.1

        # 无积分号 -> 完成
        if not state.expr.has(Integral):
            if self.initial_complexity is not None:
                final_reduction = (self.initial_complexity - state.ast_complexity()) / max(self.initial_complexity, 1)
                final_bonus = self.final_bonus * max(0, final_reduction)
            else:
                final_bonus = 0.0
            return True, self.final_bonus + final_bonus

        return False, 0.0

    def legal_actions(self, state: IntegrationState) -> List[Action]:
        """
        返回当前状态下所有合法动作（规则）。
        通过遍历所有积分原子和所有规则，尝试应用规则，若返回不为None则合法。
        """
        expr = state.expr
        integrals = list(expr.atoms(Integral))
        actions = set()
        for integral in integrals:
            for rule_name in get_all_rule_names():
                rule_func = get_rule_by_name(rule_name)
                try:
                    res = rule_func(integral)
                    if res is not None:
                        rule_id = self._rule_name_to_id[rule_name]
                        actions.add(Action(id=rule_id, name=rule_name))
                except Exception:
                    continue
        return list(actions)

    def step(self, state: IntegrationState, action: Action) -> Tuple[IntegrationState, float, bool, Dict[str, Any]]:
        """
        执行动作，返回 (next_state, reward, done, info)
        """
        self.steps_taken += 1
        rule_name = action.name
        rule_func = get_rule_by_name(rule_name)
        expr = state.expr

        # 查找可以应用该规则的积分项
        target_integral = None
        result = None
        for integral in expr.atoms(Integral):
            try:
                res = rule_func(integral)
                if res is not None:
                    target_integral = integral
                    result = res
                    break
            except Exception:
                continue

        # 无效动作：无法应用规则
        if target_integral is None or result is None:
            # 返回原状态，给予惩罚，并终止（或可设为不终止？通常无效动作视为非法，但为了训练，我们让其终止并给负奖励）
            # 这里选择终止该回合，避免无意义循环
            return state, -self.invalid_action_penalty, True, {"msg": "invalid_action"}

        new_sub_expr, status = result
        if status == "substitution":
            u_sym = Symbol('u')
            # 处理换元：将积分变量换回 x
            temp_integral = new_sub_expr['integral'].subs(u_sym, Symbol('x'))
            new_expr = expr.subs(target_integral, new_sub_expr['factor'] * temp_integral)
        else:
            new_expr = expr.subs(target_integral, new_sub_expr)

        old_c = state.ast_complexity()
        next_state = IntegrationState(
            expr=new_expr,
            depth=state.depth + 1,
            history_hashes=state.history_hashes.union({state.canonical_hash()})
        )
        new_c = next_state.ast_complexity()

        # 计算化简奖励
        simplify_reward = self._compute_simplify_reward(old_c, new_c)
        # 步骤惩罚
        step_cost = -self.step_penalty

        # 判断是否终止并获得终止奖励
        done, terminal_reward = self.is_terminal(next_state)
        if done:
            reward = terminal_reward   # 终止奖励已包含最终奖励或部分奖励
        else:
            reward = simplify_reward + step_cost

        info = {
            "simplify_reward": simplify_reward,
            "step_penalty": step_cost,
            "complexity_before": old_c,
            "complexity_after": new_c,
            "steps": self.steps_taken,
            "time_elapsed": time.time() - self.start_time if self.start_time else 0.0
        }
        return next_state, reward, done, info
