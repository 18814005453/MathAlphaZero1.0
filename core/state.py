# core/state.py
import hashlib
from sympy import preorder_traversal
from dataclasses import dataclass, field
from typing import Set, Optional

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
        from sympy import srepr
        return int(hashlib.md5(srepr(self.expr).encode()).hexdigest(), 16)

    def ast_complexity(self) -> int:
        """计算 AST 节点总数（带缓存）"""
        if self._ast_complexity is None:
            # 计算节点数
            self._ast_complexity = sum(1 for _ in preorder_traversal(self.expr))
        return self._ast_complexity
