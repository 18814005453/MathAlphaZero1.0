# utils/preprocessor.py
import torch
import sympy as sp
import re
from typing import List, Dict, Tuple, Optional, Any
from sympy.core.basic import Basic
from sympy.core.symbol import Symbol
from sympy.core.numbers import Number
from sympy.core.function import FunctionClass, AppliedUndef
from sympy import Integral, Derivative, Add, Mul, Pow

class MathPreprocessor:
    """
    符号积分预处理器：规范化表达式、分词、编码为固定长度张量。
    升级特性：
    - 支持精确的 AST 遍历，生成 token 序列并记录每个 token 对应的 SymPy 节点
    - 支持括号深度/AST 层级深度嵌入
    - 支持批处理
    """
    def __init__(self, max_len: int = 128, canonicalize_depth: int = 2):
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1
        self.canonicalize_depth = canonicalize_depth
        self._build_vocab()

    def _build_vocab(self):
        """构建词表：特殊符号、运算符、变量、常数、函数、其他"""
        specials = ['[PAD]', '[UNK]']
        operators = ['+', '-', '*', '/', '**', '(', ')', ',', '=', '[', ']']
        variables = ['x', 'y', 'z', 't', 'u', 'v', 'w']
        constants = ['pi', 'E', 'I', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        functions = [
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
            'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
            'asinh', 'acosh', 'atanh',
            'exp', 'log', 'ln', 'sqrt', 'Integral', 'Derivative', 'diff'
        ]
        others = ['Abs', 'sign', 'floor', 'ceiling', 'gamma']
        all_tokens = specials + operators + variables + constants + functions + others
        unique = []
        for tok in all_tokens:
            if tok not in unique:
                unique.append(tok)
        self.token2id = {tok: i for i, tok in enumerate(unique)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        assert self.token2id['[PAD]'] == 0
        assert self.token2id['[UNK]'] == 1

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ---------- 规范化 ----------
    def _canonicalize_expr(self, expr: sp.Expr, depth: int = 0) -> sp.Expr:
        if depth > self.canonicalize_depth:
            return expr
        expr = sp.expand(expr)
        expr = sp.cancel(expr)
        try:
            expr = sp.trigsimp(expr, deep=True)
        except Exception:
            pass
        if expr.is_Atom:
            return expr
        new_args = [self._canonicalize_expr(arg, depth+1) for arg in expr.args]
        expr = expr.func(*new_args)

        if expr.is_Add:
            terms = expr.as_ordered_terms(order='lex')
            expr = sp.Add(*terms, evaluate=False)
        elif expr.is_Mul:
            coeff, factors = expr.as_coeff_mul()
            if coeff != 1:
                factors = (coeff,) + factors
            sorted_factors = sorted(factors, key=lambda x: (0 if x.is_number else 1, str(x)))
            expr = sp.Mul(*sorted_factors, evaluate=False)
        elif expr.is_Pow:
            base = expr.base
            exp = expr.exp
            if exp == 1:
                expr = base
            elif exp == 0:
                expr = sp.Integer(1)
        return expr

    def standardize(self, expr: sp.Expr) -> sp.Expr:
        if expr is None:
            return sp.Integer(0)
        try:
            return self._canonicalize_expr(expr)
        except Exception:
            return expr

    # ---------- AST 遍历生成 token 序列及节点映射 ----------
    def _traverse_expr(self, expr: sp.Expr, tokens: List[str], nodes: List[Optional[sp.Expr]], depth: int, depth_list: List[int]):
        """
        递归遍历表达式，生成 token 序列，并记录每个 token 对应的节点（当前 expr）和当前深度。
        策略：
        - 对于原子节点 (Symbol, Number)，输出其名称，节点为 expr。
        - 对于函数调用 (sin, cos, Integral...)，输出函数名，然后 '('，递归参数，最后 ')'。
        - 对于加法 Add，按顺序输出每个子项，并在项之间输出 '+'（简化：假设总是输出所有项并用 '+' 连接）。
        - 对于乘法 Mul，输出因子，因子间输出 '*'（同加法处理）。
        - 对于幂 Pow，输出 base，然后 '**', exp。
        - 对于括号，我们实际上不需要显式括号，因为函数调用自带了括号。但为了统一，我们不在普通表达式中加括号。
        """
        # 记录当前节点对应的深度（即当前表达式的嵌套深度）
        depth_list.append(depth)
        if isinstance(expr, Symbol):
            tokens.append(str(expr))
            nodes.append(expr)
        elif isinstance(expr, Number):
            # 数字转字符串，注意整数和浮点数
            if expr.is_Integer:
                tokens.append(str(int(expr)))
            else:
                tokens.append(str(expr))
            nodes.append(expr)
        elif isinstance(expr, Integral):
            # Integral(expr, (var, a, b)?) 简化为 Integral(expr, var) 形式
            tokens.append('Integral')
            nodes.append(expr)
            # 左括号
            tokens.append('(')
            nodes.append(expr)
            # 被积表达式
            self._traverse_expr(expr.args[0], tokens, nodes, depth+1, depth_list)
            # 积分变量
            if len(expr.args) == 2 and isinstance(expr.args[1], sp.Tuple):
                var = expr.args[1].args[0]
                tokens.append(',')
                nodes.append(expr)
                self._traverse_expr(var, tokens, nodes, depth+1, depth_list)
            else:
                # 简单情况 Integral(f, x)
                tokens.append(',')
                nodes.append(expr)
                self._traverse_expr(expr.args[1], tokens, nodes, depth+1, depth_list)
            tokens.append(')')
            nodes.append(expr)
        elif isinstance(expr, (sp.FunctionClass, AppliedUndef)):
            # 普通函数如 sin, cos, exp 等
            func_name = expr.func.__name__
            tokens.append(func_name)
            nodes.append(expr)
            tokens.append('(')
            nodes.append(expr)
            for i, arg in enumerate(expr.args):
                if i > 0:
                    tokens.append(',')
                    nodes.append(expr)
                self._traverse_expr(arg, tokens, nodes, depth+1, depth_list)
            tokens.append(')')
            nodes.append(expr)
        elif isinstance(expr, Add):
            # 加法: 项 + 项 + ...
            args = expr.args
            for i, arg in enumerate(args):
                self._traverse_expr(arg, tokens, nodes, depth+1, depth_list)
                if i < len(args)-1:
                    tokens.append('+')
                    nodes.append(expr)
        elif isinstance(expr, Mul):
            # 乘法: 因子 * 因子 * ...
            args = expr.args
            for i, arg in enumerate(args):
                self._traverse_expr(arg, tokens, nodes, depth+1, depth_list)
                if i < len(args)-1:
                    tokens.append('*')
                    nodes.append(expr)
        elif isinstance(expr, Pow):
            self._traverse_expr(expr.base, tokens, nodes, depth+1, depth_list)
            tokens.append('**')
            nodes.append(expr)
            self._traverse_expr(expr.exp, tokens, nodes, depth+1, depth_list)
        elif isinstance(expr, sp.Tuple):
            tokens.append('(')
            nodes.append(expr)
            for i, arg in enumerate(expr.args):
                if i > 0:
                    tokens.append(',')
                    nodes.append(expr)
                self._traverse_expr(arg, tokens, nodes, depth+1, depth_list)
            tokens.append(')')
            nodes.append(expr)
        else:
            # 其他未知类型，转为字符串（可能丢失信息，但 fallback）
            s = str(expr)
            # 简单分词
            parts = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|[+\-*/%^&|~!<>=@$?:]+|[()\[\]{}.,;]', s)
            for p in parts:
                tokens.append(p)
                nodes.append(expr)

    def ast_to_token_sequence_with_nodes(self, expr: sp.Expr) -> Tuple[List[str], List[Optional[sp.Expr]], List[int]]:
        """
        返回三元组 (tokens, nodes, depths)
        - tokens: token 字符串列表
        - nodes: 每个 token 对应的 SymPy 节点（表达式对象），None 可能对于括号等（但这里我们为所有 token 关联了节点）
        - depths: 每个 token 对应的 AST 深度（括号嵌套层数）
        """
        expr = self.standardize(expr)
        tokens = []
        nodes = []
        depths = []
        self._traverse_expr(expr, tokens, nodes, 0, depths)
        return tokens, nodes, depths

    def get_bracket_depth(self, expr: sp.Expr) -> List[int]:
        """返回每个 token 的括号深度（AST 深度），兼容原接口"""
        _, _, depths = self.ast_to_token_sequence_with_nodes(expr)
        return depths

    # ---------- 编码 ----------
    def encode(self, expr: sp.Expr) -> torch.Tensor:
        """返回形状 (1, max_len) 的 LongTensor"""
        tokens, _, _ = self.ast_to_token_sequence_with_nodes(expr)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def state_to_tensor(self, expr: sp.Expr) -> torch.Tensor:
        """与 MCTS 引擎对接的接口，等同于 encode"""
        return self.encode(expr)

    def state_to_tensor_with_depth(self, expr: sp.Expr) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (token_tensor, depth_tensor)，形状均为 (1, max_len)"""
        tokens, _, depths = self.ast_to_token_sequence_with_nodes(expr)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            depths = depths[:self.max_len]
        else:
            pad_len = self.max_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            depths = depths + [0] * pad_len
        token_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        depth_tensor = torch.tensor(depths, dtype=torch.long).unsqueeze(0)
        return token_tensor, depth_tensor

    def _string_to_ids(self, s: str) -> torch.Tensor:
        """将规则名称等字符串转换为 token ID 序列（1D 张量）"""
        # 简单分词：按字母数字和分隔符
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|[+\-*/%^&|~!<>=@$?:]+|[()\[\]{}.,;]', s)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def tokenize_list(self, string_list: List[str]) -> torch.Tensor:
        if not string_list:
            return torch.empty((0, self.max_len), dtype=torch.long)
        tensor_list = [self._string_to_ids(s) for s in string_list]
        return torch.stack(tensor_list, dim=0)

    def decode(self, tensor: torch.Tensor) -> str:
        ids = tensor.squeeze(0).tolist() if tensor.dim() == 2 else tensor.tolist()
        tokens = [self.id2token.get(i, '[UNK]') for i in ids if i != self.pad_id]
        return ' '.join(tokens)