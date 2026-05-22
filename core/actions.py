# core/actions.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class Action:
    """
    AlphaZero 符号积分动作空间实例。
    使用 frozen=True 自动生成完美的 __hash__ 和 __eq__，
    满足 MCTS 树搜索节点在 Dict/Set 中 O(1) 复杂度的查重和映射。

    升级特性：
    - 支持序列化（to_dict / from_dict）
    - 可附加优先级权重（用于优先经验回放）
    - 更丰富的表示
    """
    id: int
    name: str

    def __repr__(self):
        return f"{self.name} (id={self.id})"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，便于存储经验池"""
        return {
            "id": self.id,
            "name": self.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """从字典反序列化"""
        return cls(id=data["id"], name=data["name"])

    def priority(self, base_priority: float = 1.0, **kwargs) -> float:
        """
        返回动作的优先级，用于优先经验回放。
        基类返回固定值，子类或外部可根据规则使用频率、历史成功率等动态计算。
        """
        return base_priority