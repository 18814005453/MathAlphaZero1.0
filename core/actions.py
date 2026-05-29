# core/actions.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class Action:
    """
    AlphaZero 符号积分动作空间实例（升级版：包含规则ID、规则名和位置索引）。
    使用 frozen=True 自动生成完美的 __hash__ 和 __eq__，
    满足 MCTS 树搜索节点在 Dict/Set 中 O(1) 复杂度的查重和映射。

    升级特性：
    - 新增 pos 属性，定位 Token 序列中的操作目标位置
    - 支持序列化（to_dict / from_dict）
    - 可附加优先级权重（用于优先经验回放）
    """
    id: int          # 规则 ID
    name: str        # 规则名称
    pos: int         # Token 序列中的位置索引 (0-based)

    def __repr__(self):
        return f"{self.name} (id={self.id}) @ pos={self.pos}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，便于存储经验池"""
        return {
            "id": self.id,
            "name": self.name,
            "pos": self.pos
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """从字典反序列化"""
        return cls(id=data["id"], name=data["name"], pos=data["pos"])

    def priority(self, base_priority: float = 1.0, **kwargs) -> float:
        """
        返回动作的优先级，用于优先经验回放。
        基类返回固定值，子类或外部可根据规则使用频率、历史成功率等动态计算。
        """
        return base_priority