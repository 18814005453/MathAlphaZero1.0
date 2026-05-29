# core/network.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable, Optional, Dict, Tuple

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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RelativePositionBias(nn.Module):
    """相对位置偏置（用于 Transformer 自注意力）"""
    def __init__(self, num_heads: int, max_len: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.relative_bias = nn.Parameter(torch.zeros(1, num_heads, max_len, max_len))

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.relative_bias[:, :, :seq_len, :seq_len]

class MaskedAvgPool(nn.Module):
    """带掩码的平均池化"""
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_key_padding_mask
        mask = mask.unsqueeze(-1).float()
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1.0)
        return sum_x / valid_len

class TransformerEncoderWithResidual(nn.Module):
    """Transformer编码器层，包含自注意力、前馈网络、残差连接和层归一化"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, use_rel_pos: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_bias = RelativePositionBias(nhead)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = src.size(1)
        attn_mask = None
        if self.use_rel_pos:
            attn_mask = self.rel_pos_bias(seq_len).to(src.device)
        src2, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=attn_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class MathNet(nn.Module):
    """
    双塔 + 动作定位头 (Rule + Location) 网络。
    支持动态规则缓存、可学习温度、位置掩码。
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
            num_actions: Optional[int] = None,
            use_depth_embedding: bool = True,
            max_depth: int = 32
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = 0
        self.use_depth_embedding = use_depth_embedding

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        if use_depth_embedding:
            self.depth_embedding = nn.Embedding(max_depth, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout, learnable=True)
        self.rule_pos_encoder = PositionalEncoding(d_model, max_len, dropout, learnable=True)
        self.pool = MaskedAvgPool()

        # State encoder (左塔) - 输出 token-level 特征矩阵
        self.state_encoder_layers = nn.ModuleList([
            TransformerEncoderWithResidual(d_model, nhead, 4 * d_model, dropout, use_rel_pos=False)
            for _ in range(num_layers)
        ])

        # Rule encoder (右塔)
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

        # 定位头 (Location Head): 跨注意力
        self.location_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(temperature)))

        # 规则缓存
        self._rule_embeddings = torch.empty(0, d_model)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_state(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码状态，返回:
        - state_vec: (batch, d_model) 全局池化向量
        - H_state:   (batch, seq_len, d_model) token 级别特征矩阵
        """
        mask = (x == self.pad_id)  # (batch, seq_len)
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False

        emb = self.embedding(x) * math.sqrt(self.d_model)

        if self.use_depth_embedding and depth is not None:
            # 确保 depth 序列长度与 x 一致
            if depth.size(1) != x.size(1):
                if depth.size(1) < x.size(1):
                    # 填充 pad_id (0) 到右侧
                    pad = torch.zeros(depth.size(0), x.size(1) - depth.size(1), dtype=depth.dtype, device=depth.device)
                    depth = torch.cat([depth, pad], dim=1)
                else:
                    # 截断到 x 的长度
                    depth = depth[:, :x.size(1)]
            depth_emb = self.depth_embedding(depth.clamp(max=self.depth_embedding.num_embeddings-1))
            emb = emb + depth_emb

        emb = self.pos_encoder(emb)

        for layer in self.state_encoder_layers:
            emb = layer(emb, src_key_padding_mask=mask)

        H_state = emb
        state_vec = self.pool(H_state, mask)   # (batch, d_model)
        return state_vec, H_state

    def _encode_rules(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        mask = (rule_tokens == self.pad_id)
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False
        emb = self.embedding(rule_tokens) * math.sqrt(self.d_model)
        emb = self.rule_pos_encoder(emb)
        for layer in self.rule_encoder_layers:
            emb = layer(emb, src_key_padding_mask=mask)
        rule_vecs = self.pool(emb, mask)
        return rule_vecs

    def refresh_rule_cache(self, rule_texts: List[str], tokenizer_fn: Callable, action_ids: Optional[List[int]] = None):
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

        token_list = []
        for rule in rule_texts:
            tokens = tokenizer_fn(rule)
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            else:
                tokens = tokens.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
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

    def get_rule_embeddings(self) -> torch.Tensor:
        return self._rule_embeddings

    @property
    def rule_cache_valid(self) -> bool:
        return getattr(self, '_rule_embeddings', None) is not None and self._rule_embeddings.size(0) > 0

    def _get_rule_logits(self, state_vec: torch.Tensor) -> torch.Tensor:
        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。请先调用 refresh_rule_cache")
        rule_emb = self._rule_embeddings.to(state_vec.device)
        state_norm = F.normalize(state_vec, p=2, dim=-1)
        cosine_sim = torch.matmul(state_norm, rule_emb.T)
        temperature = torch.exp(self.log_temperature.clamp(max=math.log(100.0)))
        return cosine_sim / temperature

    def _get_location_logits(self, H_state: torch.Tensor, rule_vec: torch.Tensor, location_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        rule_vec_unsqueezed = rule_vec.unsqueeze(1)
        attn_out, attn_weights = self.location_cross_attn(
            query=rule_vec_unsqueezed,
            key=H_state,
            value=H_state,
            key_padding_mask=None
        )
        location_logits = torch.matmul(H_state, rule_vec.unsqueeze(-1)).squeeze(-1)
        if location_mask is not None:
            location_logits = location_logits.masked_fill(~location_mask, -1e9)
        return location_logits

    def forward(
        self,
        x: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        rule_mask: Optional[torch.Tensor] = None,
        location_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_vec, H_state = self._encode_state(x, depth)
        value = self.value_head(state_vec)

        rule_logits = self._get_rule_logits(state_vec)
        if rule_mask is not None:
            rule_logits = rule_logits.masked_fill(~rule_mask, -1e9)

        location_logits = self._get_location_logits(H_state, state_vec, location_mask)
        return rule_logits, location_logits, value

    def predict_rule_and_location(
        self,
        x: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        rule_mask: Optional[torch.Tensor] = None,
        location_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_vec, H_state = self._encode_state(x, depth)
        value = self.value_head(state_vec)

        rule_logits = self._get_rule_logits(state_vec)
        if rule_mask is not None:
            rule_logits = rule_logits.masked_fill(~rule_mask, -1e9)
        rule_probs = F.softmax(rule_logits, dim=-1)

        rule_emb = self._rule_embeddings.to(state_vec.device)
        weighted_rule_vec = torch.matmul(rule_probs.unsqueeze(1), rule_emb).squeeze(1)
        location_logits = self._get_location_logits(H_state, weighted_rule_vec, location_mask)
        location_probs = F.softmax(location_logits, dim=-1)

        return rule_probs, location_probs, value

# 兼容原命名
MathAlphaZeroNet = MathNet