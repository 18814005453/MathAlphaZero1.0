from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    id: int
    name: str

    def __repr__(self):
        return f"Action({self.name})"

# 这里定义了你的动作空间，必须与你在 network 中设置的 num_actions 一致
# 假设你有 20 条规则，我们就定义 20 个 Action
from core.rules import RULE_NAMES

# 将规则名映射为 Action 对象
ACTIONS = [Action(i, name) for i, name in enumerate(RULE_NAMES)]
NUM_ACTIONS = len(ACTIONS)

def get_action_by_id(action_id):
    return ACTIONS[action_id]