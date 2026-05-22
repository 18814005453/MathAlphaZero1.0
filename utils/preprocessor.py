# utils/preprocessor.py
import torch
import sympy as sp
import re
from typing import List, Dict, Tuple, Optional
from sympy.core.sympify import SympifyError

class MathPreprocessor:
    """
    符号积分预处理器：规范化表达式、分词、编码为固定长度张量。
    升级特性：
    - 基于 sympy 的深度规范化：expand, cancel, trigsimp, 排序加法乘法项
    - 稳健的分词器支持数字、函数名、运算符、常量
    - 支持批处理 tokenize_list
    - 未知符号映射为 CONSTANT
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
        # 确保特殊 ID 正确
        assert self.token2id['[PAD]'] == 0
        assert self.token2id['[UNK]'] == 1

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ---------- 规范化核心 ----------
    def _canonicalize_expr(self, expr: sp.Expr, depth: int = 0) -> sp.Expr:
        """递归规范化表达式：展开、约分、三角简化、排序"""
        if depth > self.canonicalize_depth:
            return expr
        # 基础简化
        expr = sp.expand(expr)
        expr = sp.cancel(expr)
        try:
            expr = sp.trigsimp(expr, deep=True)
        except Exception:
            pass
        # 对子表达式递归
        if expr.is_Atom:
            return expr
        new_args = [self._canonicalize_expr(arg, depth+1) for arg in expr.args]
        expr = expr.func(*new_args)

        # 加法项排序
        if expr.is_Add:
            terms = expr.as_ordered_terms(order='lex')
            expr = sp.Add(*terms, evaluate=False)
        # 乘法因子排序：常数优先，然后按字符串
        elif expr.is_Mul:
            coeff, factors = expr.as_coeff_mul()
            if coeff != 1:
                factors = (coeff,) + factors
            sorted_factors = sorted(factors, key=lambda x: (0 if x.is_number else 1, str(x)))
            expr = sp.Mul(*sorted_factors, evaluate=False)
        # 幂简化
        elif expr.is_Pow:
            base = expr.base
            exp = expr.exp
            if exp == 1:
                expr = base
            elif exp == 0:
                expr = sp.Integer(1)
        return expr

    def standardize(self, expr: sp.Expr) -> sp.Expr:
        """对外标准化接口"""
        if expr is None:
            return sp.Integer(0)
        try:
            return self._canonicalize_expr(expr)
        except Exception:
            return expr

    # ---------- 分词 ----------
    def _tokenize_srepr(self, s: str) -> List[str]:
        """
        将 SymPy 的 srepr 字符串分词，这是最可靠的方法。
        示例：'Integral(Mul(Symbol('x'), Integer(2)), Tuple(Symbol('x')))'
        """
        # 使用正则提取标识符、数字、括号、逗号等
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|[+\-*/%^&|~!<>=@$?:]+|[()\[\]{}.,;]', s)
        # 过滤空
        tokens = [t for t in tokens if t]
        # 转换未知标识符为 CONSTANT
        for i, tok in enumerate(tokens):
            if tok.isalpha() and tok not in self.token2id and tok not in ('x', 'y', 'z', 't', 'u', 'v', 'w'):
                # 保留变量名，但不在词表中的函数名或符号映射为 CONSTANT
                # 注意：变量名本身在词表中，所以 isalpha() 且 not in token2id 且不是常见变量名，则替换
                if tok not in ('x', 'y', 'z', 't', 'u', 'v', 'w'):
                    tokens[i] = 'CONSTANT'
        return tokens

    def expr_to_tokens(self, expr: sp.Expr) -> List[str]:
        """将 SymPy 表达式转为 token 字符串列表"""
        expr = self.standardize(expr)
        # 使用 srepr 保证结构唯一
        s = sp.srepr(expr)
        return self._tokenize_srepr(s)

    # ---------- 编码 ----------
    def encode(self, expr: sp.Expr) -> torch.Tensor:
        """返回形状 (1, max_len) 的 LongTensor"""
        tokens = self.expr_to_tokens(expr)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def state_to_tensor(self, expr: sp.Expr) -> torch.Tensor:
        """与 MCTS 引擎对接的接口，等同于 encode"""
        return self.encode(expr)

    def _string_to_ids(self, s: str) -> torch.Tensor:
        """
        将规则名称等字符串直接转换为 token ID 序列（1D 张量）。
        用于刷新规则缓存。
        """
        # 对于规则名称，它本身就是一个 token（如 "PowerRule"），直接分词
        # 但为了通用，我们仍然使用 srepr 风格的 tokenization？不，规则名称是标识符
        # 简单处理：将字符串按字母数字分割
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+', s)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def tokenize_list(self, string_list: List[str]) -> torch.Tensor:
        """
        批量处理字符串列表，返回形状 (batch, max_len)。
        用于规则缓存刷新。
        """
        if not string_list:
            return torch.empty((0, self.max_len), dtype=torch.long)
        tensor_list = [self._string_to_ids(s) for s in string_list]
        return torch.stack(tensor_list, dim=0)

    # ---------- 解码（调试用） ----------
    def decode(self, tensor: torch.Tensor) -> str:
        """将张量还原为 token 字符串（不保证可解析）"""
        ids = tensor.squeeze(0).tolist() if tensor.dim() == 2 else tensor.tolist()
        tokens = [self.id2token.get(i, '[UNK]') for i in ids if i != self.pad_id]
        return ' '.join(tokens)