# utils/preprocessor.py (升级版)
import torch
import sympy as sp
from typing import List, Dict, Tuple, Optional
from sympy.core.sympify import SympifyError


class MathPreprocessor:
    """
    符合 AlphaZero 积分架构的预处理器：
    - 固定词表（~100-200），含 [PAD], [UNK], 数学算子, 函数, 变量, 常数占位符
    - 强大的规范化（交换律排序、合并常数、消除冗余）
    - 输出 padding 固定长度 (max_len)，[PAD] ID = 0
    - 提供 state_to_tensor(expr) -> torch.LongTensor of shape (1, max_len)
    """

    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1

        # 构建完整的词表（顺序决定 ID，0 和 1 必须为 [PAD] 和 [UNK]）
        self._build_vocab()

    def _build_vocab(self):
        # 1. 特殊 token
        specials = ['[PAD]', '[UNK]']
        # 2. 基础运算符号
        operators = ['+', '-', '*', '/', '**', '(', ')']
        # 3. 变量（自变量 + 任意常数占位符）
        variables = ['x', 'y', 'CONSTANT']
        # 4. 常见常数
        constants = ['pi', 'E', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # 5. 基本初等函数（包括积分动作需要的 Integral）
        functions = [
            'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
            'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
            'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
            'exp', 'log', 'ln', 'sqrt', 'Integral'
        ]
        # 6. 额外常用符号（如微分算子等，可选）
        others = ['diff', 'Derivative']

        # 合并全部 token，去重保持顺序
        all_tokens = specials + operators + variables + constants + functions + others
        # 去重（保留第一次出现的顺序）
        unique_tokens = []
        for tok in all_tokens:
            if tok not in unique_tokens:
                unique_tokens.append(tok)

        self.token2id = {tok: idx for idx, tok in enumerate(unique_tokens)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}

        # 确保 PAD 和 UNK 的 ID 正确（0 和 1）
        assert self.token2id['[PAD]'] == 0
        assert self.token2id['[UNK]'] == 1

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    # ------------------------------------------------------------
    # 规范化（canonicalization）核心
    # ------------------------------------------------------------
    def _canonical(self, expr: sp.Expr) -> sp.Expr:
        """
        将表达式转化为规范形式，满足：
        - 加法项按确定顺序排序（常数项优先，然后按变量/函数名的字符串排序）
        - 乘法因子按确定顺序（数字系数优先，然后按符号名称排序）
        - 消除无意义的系数 1 和指数 1
        - 将有理数转化为分数形式
        - 统一用 Rational 表示常数
        """
        # 基础展开（但不是必需，可去除乘积中的括号）
        expr = sp.expand(expr, deep=False)
        # 递归处理
        return self._canonical_rec(expr)

    def _canonical_rec(self, expr: sp.Expr) -> sp.Expr:
        if expr.is_Atom:
            return expr

        # 处理加法：排序各项
        if expr.is_Add:
            args = expr.args
            # 对每个参数递归规范化
            norm_args = [self._canonical_rec(arg) for arg in args]

            # 排序：使用自定义 key (类型, 字符串表示)
            def sort_key(arg):
                # 常数排前面，然后按字符串排序
                if arg.is_number:
                    return (0, str(arg))
                return (1, str(arg))

            sorted_args = sorted(norm_args, key=sort_key)
            return sp.Add(*sorted_args, evaluate=False)

        # 处理乘法：排序因子
        if expr.is_Mul:
            args = expr.args
            norm_args = [self._canonical_rec(arg) for arg in args]
            # 分离数字系数和其他因子
            coeff = sp.Integer(1)
            others = []
            for arg in norm_args:
                if arg.is_number:
                    coeff *= arg
                else:
                    others.append(arg)
            # 对非数字因子排序（按字符串）
            others.sort(key=lambda x: str(x))
            # 构建新乘法：如果系数为1，省略系数；系数为-1，保留负号
            if coeff == 1:
                mul_args = others
            else:
                mul_args = [coeff] + others
            if not mul_args:
                return sp.Integer(1)
            if len(mul_args) == 1:
                return mul_args[0]
            return sp.Mul(*mul_args, evaluate=False)

        # 处理幂：规范化指数
        if expr.is_Pow:
            base = self._canonical_rec(expr.base)
            exp = self._canonical_rec(expr.exp)
            if exp == 1:
                return base
            return sp.Pow(base, exp, evaluate=False)

        # 处理函数：递归规范化参数
        if expr.is_Function:
            args = [self._canonical_rec(arg) for arg in expr.args]
            return expr.func(*args, evaluate=False)

        # 其他（如符号）直接返回
        return expr

    def standardize(self, expr: sp.Expr) -> sp.Expr:
        """对外公开的标准化接口"""
        if expr is None:
            return sp.Integer(0)
        try:
            return self._canonical(expr)
        except Exception:
            # 退化情况，返回原始表达式
            return expr

    # ------------------------------------------------------------
    # 表达式 → Token 序列（稳健分词 + 未知符号映射为 CONSTANT）
    # ------------------------------------------------------------
    def expr_to_tokens(self, expr: sp.Expr) -> List[str]:
        """将 SymPy 表达式转为 token 字符串列表"""
        expr = self.standardize(expr)
        # 使用 sympy 的 srepr? 不，简单字符串分词更可控
        s = str(expr)
        return self._tokenize_string(s)

    def _tokenize_string(self, s: str) -> List[str]:
        """对字符串进行数学感知的分词"""
        tokens = []
        i = 0
        n = len(s)
        while i < n:
            # 处理两位运算符
            if i + 1 < n and s[i:i + 2] in ('**', '//', '==', '!='):
                tokens.append(s[i:i + 2])
                i += 2
                continue
            # 单字符运算符/括号
            if s[i] in '+-*/()[],':
                tokens.append(s[i])
                i += 1
                continue
            # 数字（支持整数、小数、科学计数法）
            if s[i].isdigit() or s[i] == '.':
                start = i
                while i < n and (s[i].isdigit() or s[i] == '.' or s[i] == 'e' or s[i] == 'E' or (
                        s[i] in '+-' and i > start and s[i - 1].lower() == 'e')):
                    i += 1
                token = s[start:i]
                # 将数字标准化：去掉前导零等，但保留为字符串
                # 为了方便，可以直接保留原始字符串
                tokens.append(token)
                continue
            # 字母开头的标识符（函数名、变量名、常数名）
            if s[i].isalpha():
                start = i
                while i < n and (s[i].isalpha() or s[i].isdigit()):
                    i += 1
                token = s[start:i]
                # 将未知标识符映射为 CONSTANT
                if token not in self.token2id:
                    # 如果是常用数学常数但未收录？优先检查
                    if token in ('Pi', 'pi'):
                        token = 'pi'
                    elif token in ('E', 'e'):
                        token = 'E'
                    else:
                        token = 'CONSTANT'
                tokens.append(token)
                continue
            # 其他字符（空格等跳过）
            i += 1
        return tokens

    # ------------------------------------------------------------
    # 编码：Token 序列 → 固定长度张量
    # ------------------------------------------------------------
    def encode(self, expr: sp.Expr) -> torch.Tensor:
        """
        返回形状 (1, max_len) 的 LongTensor，可直接输入网络
        """
        tokens = self.expr_to_tokens(expr)
        ids = [self.token2id.get(tok, self.unk_id) for tok in tokens]
        # 截断或填充
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, max_len]

    def state_to_tensor(self, expr: sp.Expr) -> torch.Tensor:
        """
        与 MCTS 引擎对接的标准接口，等同于 encode
        """
        return self.encode(expr)

    # ------------------------------------------------------------
    # 辅助：张量 → 表达式（用于调试）
    # ------------------------------------------------------------
    def decode(self, tensor: torch.Tensor) -> str:
        """将张量还原为 token 字符串序列（不保证可解析为 SymPy 表达式）"""
        ids = tensor.squeeze(0).tolist() if tensor.dim() == 2 else tensor.tolist()
        tokens = [self.id2token.get(i, '[UNK]') for i in ids if i != self.pad_id]
        return ' '.join(tokens)