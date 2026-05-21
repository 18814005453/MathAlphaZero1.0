import os
import re

print("🔧 开始完整修复...")

# 1. 完全重写 env.py
env_content = '''# core/env.py
from sympy import Integral, preorder_traversal, Symbol
from core.state import IntegrationState
from knowledge.rule_registry import get_rule_by_name, get_rule_id, get_all_rule_names, get_num_rules
from core.actions import Action
import time
from typing import Tuple, Optional, Dict, Any

class IntegrationEnv:
    def __init__(self, max_steps=20, time_limit=None, simplify_reward_alpha=0.2, simplify_reward_beta=3.0):
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.simplify_alpha = simplify_reward_alpha
        self.simplify_beta = simplify_reward_beta
        self.steps_taken = 0
        self.start_time = None
        self.initial_complexity = None
        self._rule_name_to_id = {name: idx for idx, name in enumerate(get_all_rule_names())}
        self._valid_rule_ids = set(self._rule_name_to_id.values())

    def reset(self, expr):
        self.steps_taken = 0
        self.start_time = time.time() if self.time_limit else None
        state = IntegrationState(expr=expr, depth=0, history_hashes=set())
        self.initial_complexity = state.ast_complexity()
        return state

    def _compute_simplify_reward(self, old_complexity: int, new_complexity: int) -> float:
        if new_complexity <= old_complexity:
            reduction_ratio = (old_complexity - new_complexity) / max(old_complexity, 1)
            r = self.simplify_alpha * (reduction_ratio ** 0.5)
            return min(r, 0.3)
        else:
            increase_ratio = (new_complexity - old_complexity) / max(old_complexity, 1)
            return -0.05 * min(increase_ratio, 2.0)

    def is_terminal(self, state) -> Tuple[bool, float]:
        if self.steps_taken >= self.max_steps:
            return True, -0.3
        
        if self.time_limit is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit:
                current_c = state.ast_complexity()
                if self.initial_complexity is not None:
                    reduction = (self.initial_complexity - current_c) / max(self.initial_complexity, 1)
                    partial_reward = 0.2 * max(0, reduction)
                else:
                    partial_reward = 0.0
                return True, partial_reward - 0.1
        
        if not state.expr.has(Integral):
            if self.initial_complexity is not None:
                final_reduction = (self.initial_complexity - state.ast_complexity()) / max(self.initial_complexity, 1)
                final_bonus = 0.2 * max(0, final_reduction)
            else:
                final_bonus = 0.0
            return True, 1.0 + final_bonus
        
        return False, 0.0

    def legal_actions(self, state):
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

    def step(self, state, action):
        self.steps_taken += 1
        rule_name = action.name
        rule_func = get_rule_by_name(rule_name)
        expr = state.expr

        target_integral = None
        result = None
        for integral in expr.atoms(Integral):
            try:
                res = rule_func(integral)
                if res is not None:
                    target_integral = integral
                    result = res
                    break
            except:
                continue

        if target_integral is None or result is None:
            return state, -0.2, True, {"msg": "invalid_action"}

        new_sub_expr, status = result
        if status == "substitution":
            u_sym = Symbol('u')
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
        simplify_reward = self._compute_simplify_reward(old_c, new_c)

        done, terminal_reward = self.is_terminal(next_state)
        if done:
            reward = terminal_reward
        else:
            reward = simplify_reward

        info = {
            "simplify_reward": simplify_reward,
            "complexity_before": old_c,
            "complexity_after": new_c,
            "steps": self.steps_taken,
            "time_elapsed": time.time() - self.start_time if self.start_time else 0.0
        }
        return next_state, reward, done, info
'''

with open("core/env.py", "w") as f:
    f.write(env_content)
print("✅ core/env.py 已重写")

# 2. 修复 auto_train.py 中的 Poly 警告
train_file = "auto_train.py"
with open(train_file, "r") as f:
    lines = f.readlines()

new_lines = []
skip_until_next = False
for line in lines:
    if "poly_coeff = sp.Poly(random.randint(1, 3) * x + random.randint(1, 2), x)" in line:
        new_lines.append("    poly_coeff = random.randint(1, 3) * x + random.randint(1, 2)  # 修复 Poly 警告\n")
        continue
    if "return coeff * poly_coeff * base" in line:
        new_lines.append("    return coeff * poly_coeff * base\n")
        continue
    new_lines.append(line)

with open(train_file, "w") as f:
    f.writelines(new_lines)
print("✅ auto_train.py Poly 警告已修复")

# 3. 修复 engine.py 中的 env 初始化
engine_file = "core/engine.py"
with open(engine_file, "r") as f:
    content = f.read()

# 确保 env 初始化时 time_limit 被正确传递
content = content.replace(
    "self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout)",
    "self.env = IntegrationEnv(max_steps=max_depth, time_limit=timeout if timeout else None)"
)

with open(engine_file, "w") as f:
    f.write(content)
print("✅ core/engine.py 已修复")

# 4. 清理所有旧数据
import shutil
if os.path.exists("data"):
    for f in os.listdir("data"):
        file_path = os.path.join("data", f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("✅ 旧数据已清理")

print("\n" + "="*50)
print("✅ 所有修复完成！")
print("="*50)
print("\n现在运行: python auto_train.py\n")
