# core/state.py
import hashlib
import copy
from typing import List, Optional, Set, Dict, Any, Tuple
from sympy import preorder_traversal, srepr, Expr
from sympy.core.basic import Basic
from sympy.core.symbol import Symbol
from sympy.core.numbers import Number
from sympy.core.function import FunctionClass
from dataclasses import dataclass, field

# 为了不引起循环导入，使用 TYPE_CHECKING 或延迟导入
# 这里先定义一个 Protocol 或者直接使用预处理器类
class PreprocessorProtocol:
    def ast_to_token_sequence_with_nodes(self, expr: Expr) -> Tuple[List[str], List[Optional[Expr]], List[int]]:
        ...  # 返回 (tokens, nodes, depths)

    def get_bracket_depth(self, expr: Expr) -> List[int]:
        ...

# 全局预处理器占位，实际会在 __init__ 中注入
_DEFAULT_PREPROCESSOR = None

def set_default_preprocessor(preprocessor):
    global _DEFAULT_PREPROCESSOR
    _DEFAULT_PREPROCESSOR = preprocessor


@dataclass
class IntegrationState:
    expr: Expr
    depth: int = 0
    history_hashes: Optional[Set[int]] = field(default_factory=set)
    _ast_complexity: Optional[int] = field(default=None, init=False, repr=False)
    _token_seq: Optional[List[str]] = field(default=None, init=False, repr=False)
    _node_at_pos: Optional[List[Optional[Expr]]] = field(default=None, init=False, repr=False)
    _depth_seq: Optional[List[int]] = field(default=None, init=False, repr=False)
    _preprocessor: Optional[PreprocessorProtocol] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.history_hashes is None:
            self.history_hashes = set()
        # 初始化 token 相关缓存（延迟构建，首次访问时填充）
        self._token_seq = None
        self._node_at_pos = None
        self._depth_seq = None
        self._preprocessor = None

    def _ensure_caches(self, preprocessor=None):
        """确保 token 序列、节点映射、深度序列已生成"""
        if self._token_seq is not None:
            return
        if preprocessor is None:
            global _DEFAULT_PREPROCESSOR
            preprocessor = _DEFAULT_PREPROCESSOR
        if preprocessor is None:
            raise RuntimeError("未设置预处理器，请在 state 中提供 preprocessor 或通过 set_default_preprocessor 设置全局预处理器")
        self._preprocessor = preprocessor
        tokens, nodes, depths = preprocessor.ast_to_token_sequence_with_nodes(self.expr)
        self._token_seq = tokens
        self._node_at_pos = nodes
        self._depth_seq = depths

    @property
    def token_sequence(self) -> List[str]:
        """返回 Token 字符串列表（与网络输入一致）"""
        self._ensure_caches()
        return self._token_seq

    @property
    def token_depth_sequence(self) -> List[int]:
        """返回每个 Token 的括号深度（AST 层级深度）"""
        self._ensure_caches()
        return self._depth_seq

    def get_subexpression_at_pos(self, pos: int) -> Optional[Expr]:
        """返回该位置对应的 SymPy 子表达式（节点）"""
        self._ensure_caches()
        if 0 <= pos < len(self._node_at_pos):
            return self._node_at_pos[pos]
        return None

    def replace_subexpression_at_pos(self, pos: int, new_subexpr: Expr) -> Expr:
        """
        将原表达式中位置 pos 对应的子表达式替换为 new_subexpr，返回新表达式。
        注意：原表达式不变，返回新表达式对象。
        """
        self._ensure_caches()
        old_node = self.get_subexpression_at_pos(pos)
        if old_node is None:
            raise ValueError(f"位置 {pos} 没有对应的节点")
        # 使用 xreplace 进行替换
        new_expr = self.expr.xreplace({old_node: new_subexpr})
        return new_expr

    def canonical_hash(self) -> int:
        """使用 SymPy 的 srepr 保证结构相同即 hash 相同（无视变量名）"""
        return int(hashlib.md5(srepr(self.expr).encode()).hexdigest(), 16)

    def ast_complexity(self) -> int:
        """计算 AST 节点总数（带缓存）"""
        if self._ast_complexity is None:
            self._ast_complexity = sum(1 for _ in preorder_traversal(self.expr))
        return self._ast_complexity

    def clone(self, preprocessor=None) -> "IntegrationState":
        """深度克隆状态，保留缓存的 token 信息（浅拷贝结构可复用）"""
        from sympy import sympify
        new_state = IntegrationState(
            expr=sympify(self.expr),   # 深拷贝表达式
            depth=self.depth,
            history_hashes=set(self.history_hashes),
            _ast_complexity=self._ast_complexity
        )
        # 如果原状态已有缓存的 token 信息，新状态可以共享（因为表达式是深拷贝但节点对象不同，不能共享！）
        # 为了保证正确性，新状态需要重新生成缓存，或者我们深层复制节点列表？更简单：不复制缓存，新状态延迟生成。
        # 但为了效率，可以复制 token 字符串列表和深度列表，但节点列表是 Expression 对象的引用，深拷贝后节点不同，无法复用。
        # 因此新状态不复制缓存，留待 __post_init__ 为空。
        return new_state

    def __deepcopy__(self, memo):
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
        return (self.canonical_hash() == other.canonical_hash() and
                self.depth == other.depth and
                self.history_hashes == other.history_hashes)

    def __hash__(self):
        return hash((self.canonical_hash(), self.depth))

    def to_dict(self) -> Dict[str, Any]:
        from sympy import srepr
        return {
            "expr_srepr": srepr(self.expr),
            "depth": self.depth,
            "history_hashes": list(self.history_hashes),
            "ast_complexity": self._ast_complexity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], preprocessor=None) -> "IntegrationState":
        from sympy import parse_expr
        expr = parse_expr(data["expr_srepr"])
        state = cls(
            expr=expr,
            depth=data["depth"],
            history_hashes=set(data["history_hashes"])
        )
        state._ast_complexity = data.get("ast_complexity")
        if preprocessor:
            state._ensure_caches(preprocessor)
        return state

    def __repr__(self):
        return f"IntegrationState(expr={self.expr}, depth={self.depth}, history_len={len(self.history_hashes)})"