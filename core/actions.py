# core/actions.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    """
    AlphaZero 符号积分动作空间实例。
    使用 frozen=True 自动生成完美的 __hash__ 和 __eq__，
    满足 MCTS 树搜索节点在 Dict/Set 中 O(1) 复杂度的查重和映射。
    """
    id: int
    name: str

    def __repr__(self):
        # 优化打印格式，让主程序打印“步骤X: ← 应用规则: PowerRule”时更干净
        return f"{self.name}"