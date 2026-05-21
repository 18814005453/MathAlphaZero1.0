# core/state.py
from dataclasses import dataclass

# ✅ 加上 frozen=True，防止 Python 装饰器静默抹除你的自定义 __hash__
@dataclass(frozen=True)
class IntegrationState:
    expr: object
    depth: int = 0
    history_hashes: set = None

    def __post_init__(self):
        # ✅ 由于设置了 frozen=True，不能直接对属性赋值，必须通过 object.__setattr__ 初始化
        if self.history_hashes is None:
            object.__setattr__(self, 'history_hashes', set())

    def canonical_hash(self):
        return hash(str(self.expr))

    def __hash__(self):
        return self.canonical_hash()

    def __eq__(self, other):
        if not isinstance(other, IntegrationState):
            return False
        return str(self.expr) == str(other.expr)
