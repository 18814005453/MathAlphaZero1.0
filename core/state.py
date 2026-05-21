from dataclasses import dataclass

@dataclass
class IntegrationState:
    expr: object
    depth: int = 0
    history_hashes: set = None

    def __post_init__(self):
        if self.history_hashes is None:
            self.history_hashes = set()

    def canonical_hash(self):
        return hash(str(self.expr))

    def __hash__(self):
        return self.canonical_hash()

    def __eq__(self, other):
        return str(self.expr) == str(other.expr)
