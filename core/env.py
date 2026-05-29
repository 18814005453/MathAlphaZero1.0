# core/env.py
import math
import time
from typing import Tuple, Optional, Dict, Any, List

from sympy import Integral, Expr

from core.state import IntegrationState
from core.actions import Action
from knowledge.rule_registry import get_rule_by_name, get_all_rule_names


class IntegrationEnv:
    """
    符号积分环境，支持精确位置定位的动作执行。
    """
    def __init__(
        self,
        max_steps: int = 20,
        time_limit: Optional[float] = None,
        simplify_reward_alpha: float = 0.3,
        simplify_reward_beta: float = 2.0,
        step_penalty: float = 0.02,
        final_bonus: float = 1.0,
        invalid_action_penalty: float = 0.2,
        partial_completion_bonus: float = 0.2
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

        self._rule_name_to_id = {name: idx for idx, name in enumerate(get_all_rule_names())}
        self._rule_id_to_name = {idx: name for name, idx in self._rule_name_to_id.items()}

    def reset(self, expr: Expr) -> IntegrationState:
        self.steps_taken = 0
        self.start_time = time.time() if self.time_limit else None
        state = IntegrationState(expr=expr, depth=0, history_hashes=set())
        self.initial_complexity = state.ast_complexity()
        return state

    def _compute_simplify_reward(self, old_complexity: int, new_complexity: int) -> float:
        if new_complexity < old_complexity:
            reduction_ratio = (old_complexity - new_complexity) / max(old_complexity, 1)
            r = self.simplify_alpha * (1 - math.exp(-self.simplify_beta * reduction_ratio))
            return min(r, 0.5)
        elif new_complexity > old_complexity:
            increase_ratio = (new_complexity - old_complexity) / max(old_complexity, 1)
            return -0.1 * min(increase_ratio, 1.0)
        else:
            return 0.0

    def is_terminal(self, state: IntegrationState) -> Tuple[bool, float]:
        if self.steps_taken >= self.max_steps:
            if self.initial_complexity is not None:
                current_c = state.ast_complexity()
                reduction = (self.initial_complexity - current_c) / max(self.initial_complexity, 1)
                partial_reward = self.partial_completion_bonus * max(0, reduction)
            else:
                partial_reward = 0.0
            return True, partial_reward - 0.1

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

        if not state.expr.has(Integral):
            if self.initial_complexity is not None:
                final_reduction = (self.initial_complexity - state.ast_complexity()) / max(self.initial_complexity, 1)
                final_bonus = self.final_bonus * max(0, final_reduction)
            else:
                final_bonus = 0.0
            return True, self.final_bonus + final_bonus

        return False, 0.0

    def legal_actions(self, state: IntegrationState) -> List[Action]:
        token_seq = state.token_sequence
        seq_len = len(token_seq)
        legal = []
        for pos in range(seq_len):
            try:
                sub_expr = state.get_subexpression_at_pos(pos)
            except Exception:
                continue
            if sub_expr is None:
                continue
            for rule_name in get_all_rule_names():
                rule_func = get_rule_by_name(rule_name)
                try:
                    res = rule_func(sub_expr)
                    if res is not None:
                        rule_id = self._rule_name_to_id[rule_name]
                        legal.append(Action(id=rule_id, name=rule_name, pos=pos))
                except Exception:
                    continue
        return legal

    def step(self, state: IntegrationState, action: Action) -> Tuple[IntegrationState, float, bool, Dict[str, Any]]:
        self.steps_taken += 1
        rule_name = action.name
        rule_func = get_rule_by_name(rule_name)
        pos = action.pos

        try:
            sub_expr = state.get_subexpression_at_pos(pos)
        except Exception:
            return state, -self.invalid_action_penalty, True, {"msg": f"invalid_position_{pos}"}

        if sub_expr is None:
            return state, -self.invalid_action_penalty, True, {"msg": f"no_subexpr_at_{pos}"}

        try:
            result = rule_func(sub_expr)
        except Exception:
            result = None

        if result is None:
            return state, -self.invalid_action_penalty, True, {"msg": f"rule_{rule_name}_not_applicable_at_{pos}"}

        new_sub_expr, status = result

        try:
            new_expr = state.replace_subexpression_at_pos(pos, new_sub_expr)
        except Exception:
            return state, -self.invalid_action_penalty, True, {"msg": "replace_failed"}

        old_c = state.ast_complexity()
        next_state = IntegrationState(
            expr=new_expr,
            depth=state.depth + 1,
            history_hashes=state.history_hashes.union({state.canonical_hash()})
        )
        new_c = next_state.ast_complexity()

        simplify_reward = self._compute_simplify_reward(old_c, new_c)
        step_cost = -self.step_penalty

        done, terminal_reward = self.is_terminal(next_state)
        if done:
            reward = terminal_reward
        else:
            reward = simplify_reward + step_cost

        info = {
            "simplify_reward": simplify_reward,
            "step_penalty": step_cost,
            "complexity_before": old_c,
            "complexity_after": new_c,
            "steps": self.steps_taken,
            "time_elapsed": time.time() - self.start_time if self.start_time else 0.0,
            "applied_rule": rule_name,
            "applied_pos": pos
        }
        return next_state, reward, done, info
