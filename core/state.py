# core/state.py
import hashlib
import copy
from sympy import preorder_traversal, srepr
from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any

@dataclass
class IntegrationState:
    expr: any
    depth: int = 0
    history_hashes: Optional[Set[int]] = field(default_factory=set)
    _ast_complexity: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.history_hashes is None:
            self.history_hashes = set()

    def canonical_hash(self) -> int:
        """使用 SymPy 的 srepr 保证结构相同即 hash 相同（无视变量名）"""
        return int(hashlib.md5(srepr(self.expr).encode()).hexdigest(), 16)

    def ast_complexity(self) -> int:
        """计算 AST 节点总数（带缓存）"""
        if self._ast_complexity is None:
            self._ast_complexity = sum(1 for _ in preorder_traversal(self.expr))
        return self._ast_complexity

    def clone(self) -> "IntegrationState":
        """深度克隆状态，用于并行搜索"""
        from sympy import sympify
        return IntegrationState(
            expr=sympify(self.expr),   # 深拷贝表达式
            depth=self.depth,
            history_hashes=set(self.history_hashes),
            _ast_complexity=self._ast_complexity  # 复用缓存，无需重算
        )

    def __deepcopy__(self, memo):
        """支持 copy.deepcopy"""
        from sympy import sympify
        new_state = IntegrationState(
            expr=sympify(self.expr),
            depth=self.depth,
            history_hashes=set(self.history_hashes),
            _ast_complexity=self._ast_complexity
        )
        memo[id(self)] = new_state
        return new_state

    def __eq__(self, other):
        if not isinstance(other, IntegrationState):
            return False
        # 比较表达式哈希、深度、历史哈希集
        return (self.canonical_hash() == other.canonical_hash() and
                self.depth == other.depth and
                self.history_hashes == other.history_hashes)

    def __hash__(self):
        # 基于规范化哈希和深度（历史哈希集可变，不参与哈希）
        # 注意：历史哈希集不参与哈希，因为它是可变的。
        # 如果需要将状态作为字典键，请使用 (canonical_hash, depth)
        return hash((self.canonical_hash(), self.depth))

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，用于存储经验池"""
        from sympy import srepr
        return {
            "expr_srepr": srepr(self.expr),
            "depth": self.depth,
            "history_hashes": list(self.history_hashes),
            "ast_complexity": self._ast_complexity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationState":
        """从字典反序列化"""
        from sympy import parse_expr
        expr = parse_expr(data["expr_srepr"])
        state = cls(
            expr=expr,
            depth=data["depth"],
            history_hashes=set(data["history_hashes"])
        )
        state._ast_complexity = data.get("ast_complexity")
        return state

    def __repr__(self):
        return f"IntegrationState(expr={self.expr}, depth={self.depth}, history_len={len(self.history_hashes)})"