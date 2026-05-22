# core/network.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Optional, Dict

class PositionalEncoding(nn.Module):
    """固定的正弦余弦位置编码（可学习版本可选）"""
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1, learnable: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learnable:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MaskedAvgPool(nn.Module):
    """带掩码的平均池化，用于聚合序列特征"""
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_key_padding_mask
        mask = mask.unsqueeze(-1).float()
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1.0)
        return sum_x / valid_len

class TransformerEncoderWithResidual(nn.Module):
    """Transformer编码器层，包含自注意力、前馈网络、残差连接和层归一化"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # FFN with residual
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class MathNet(nn.Module):
    """
    增强版数学积分网络，支持动态动作空间和可学习温度。
    """
    def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 3,
            rule_num_layers: int = 2,
            max_len: int = 128,
            dropout: float = 0.1,
            temperature: float = 0.07,
            learn_temperature: bool = False,
            num_actions: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = 0

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout, learnable=True)
        self.rule_pos_encoder = PositionalEncoding(d_model, max_len, dropout, learnable=True)
        self.pool = MaskedAvgPool()

        # State encoder stack
        self.state_encoder_layers = nn.ModuleList([
            TransformerEncoderWithResidual(d_model, nhead, 4 * d_model, dropout)
            for _ in range(num_layers)
        ])

        # Rule encoder stack
        self.rule_encoder_layers = nn.ModuleList([
            TransformerEncoderWithResidual(d_model, nhead, 4 * d_model, dropout)
            for _ in range(rule_num_layers)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )

        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(temperature)))

        # 规则缓存：嵌入矩阵和映射
        self._rule_embeddings = torch.empty(0, d_model)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, x: torch.Tensor, mask: torch.Tensor, encoder_layers: nn.ModuleList, pos_encoder: PositionalEncoding) -> torch.Tensor:
        """通用编码器：嵌入 + 位置编码 + 堆叠层 + 池化"""
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = pos_encoder(emb)
        for layer in encoder_layers:
            emb = layer(emb, src_key_padding_mask=mask)
        return self.pool(emb, mask)

    def _encode_state(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._encode(x, mask, self.state_encoder_layers, self.pos_encoder)

    def _encode_rules(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        mask = (rule_tokens == self.pad_id)
        # 避免全掩码
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False
        return self._encode(rule_tokens, mask, self.rule_encoder_layers, self.rule_pos_encoder)

    def refresh_rule_cache(self, rule_texts: List[str], tokenizer_fn: Callable, action_ids: Optional[List[int]] = None):
        """
        刷新规则缓存，支持动态动作空间。
        rule_texts: 规则对应的字符串表示（用于 tokenization）
        tokenizer_fn: 将字符串转换为 token 张量的函数
        action_ids: 可选，每个规则对应的动作 ID（必须与 rule_texts 长度相同）
        """
        device = next(self.parameters()).device
        if not rule_texts:
            self._rule_embeddings = torch.empty(0, self.d_model, device=device)
            self.id_to_idx, self.idx_to_id = {}, {}
            return {}

        if action_ids is None:
            action_ids = list(range(len(rule_texts)))
        elif len(action_ids) != len(rule_texts):
            raise ValueError("action_ids length mismatch")

        self.id_to_idx = {int(act_id): idx for idx, act_id in enumerate(action_ids)}
        self.idx_to_id = {idx: int(act_id) for idx, act_id in enumerate(action_ids)}

        # 将每个规则文本转换为张量
        token_list = []
        for rule in rule_texts:
            tokens = tokenizer_fn(rule)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            else:
                tokens = tokens.to(device)

            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            # 截断或填充至 max_len
            if tokens.size(1) > self.max_len:
                tokens = tokens[:, :self.max_len]
            elif tokens.size(1) < self.max_len:
                pad = torch.full((1, self.max_len - tokens.size(1)), self.pad_id, dtype=tokens.dtype, device=device)
                tokens = torch.cat([tokens, pad], dim=1)
            token_list.append(tokens)

        tokens = torch.cat(token_list, dim=0)
        with torch.no_grad():
            rule_vecs = self._encode_rules(tokens)
            rule_vecs = F.normalize(rule_vecs, p=2, dim=-1)
        self._rule_embeddings = rule_vecs.detach().clone()
        return self.id_to_idx

    def get_rule_index(self, action_id: int) -> int:
        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。")
        idx = self.id_to_idx.get(int(action_id))
        if idx is not None:
            return idx
        # 未知动作 ID 返回最后一个索引（通常是占位）
        return len(self._rule_embeddings) - 1 if len(self._rule_embeddings) > 0 else 0

    def get_rule_embeddings(self) -> torch.Tensor:
        return self._rule_embeddings

    @property
    def rule_cache_valid(self) -> bool:
        return getattr(self, '_rule_embeddings', None) is not None and self._rule_embeddings.size(0) > 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：
        x: (batch, seq_len)  token 序列
        mask: (batch, num_rules) 布尔掩码，True 表示合法动作
        返回: policy_logits (batch, num_rules), value (batch, 1)
        """
        pad_mask = (x == self.pad_id)
        # 确保每一行至少有一个有效位置（避免全掩码）
        if pad_mask.all(dim=1).any():
            pad_mask[pad_mask.all(dim=1), 0] = False

        state_vec = self._encode_state(x, pad_mask)          # (batch, d_model)
        value = self.value_head(state_vec)                   # (batch, 1)

        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。请先调用 refresh_rule_cache")

        # 动态获取规则嵌入矩阵（可能在不同设备之间移动）
        if self._rule_embeddings.device != x.device:
            self._rule_embeddings = self._rule_embeddings.to(x.device)

        rule_emb = self._rule_embeddings                     # (num_rules, d_model)
        state_norm = F.normalize(state_vec, p=2, dim=-1)     # (batch, d_model)
        cosine_sim = torch.matmul(state_norm, rule_emb.T)    # (batch, num_rules)

        temperature = torch.exp(self.log_temperature.clamp(max=math.log(100.0)))
        policy_logits = cosine_sim / temperature             # (batch, num_rules)

        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)

        return policy_logits, value

# 兼容原命名
MathAlphaZeroNet = MathNet
