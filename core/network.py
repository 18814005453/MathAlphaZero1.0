import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Callable, Optional, Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MaskedAvgPool(nn.Module):
    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_key_padding_mask
        mask = mask.unsqueeze(-1).float()
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1.0)
        return sum_x / valid_len

class MathNet(nn.Module):
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
        self.temperature = temperature

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.rule_pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pool = MaskedAvgPool()

        state_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(state_encoder_layer, num_layers=num_layers)

        rule_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.rule_transformer_encoder = nn.TransformerEncoder(rule_encoder_layer, num_layers=rule_num_layers)

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

        self._rule_embeddings = torch.empty(0, d_model)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_state(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        encoded = self.transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)

    def _encode_rules(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        mask = (rule_tokens == self.pad_id)
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False
        emb = self.embedding(rule_tokens) * math.sqrt(self.d_model)
        emb = self.rule_pos_encoder(emb)
        encoded = self.rule_transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)

    def refresh_rule_cache(self, rule_texts: List[str], tokenizer_fn: Callable, action_ids: Optional[List[int]] = None):
        device = next(self.parameters()).device
        if not rule_texts:
            self._rule_embeddings = torch.empty(0, self.d_model, device=device)
            self.id_to_idx, self.idx_to_id = {}, {}
            return {}
        # 如果没有传入 action_ids，则使用连续整数（兼容旧逻辑）
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

    def get_rule_index(self, action_id: int) -> int:
        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。")
        idx = self.id_to_idx.get(int(action_id))
        if idx is not None:
            return idx
        return len(self._rule_embeddings) - 1

    def get_rule_embeddings(self) -> torch.Tensor:
        return self._rule_embeddings

    @property
    def rule_cache_valid(self) -> bool:
        return getattr(self, '_rule_embeddings', None) is not None and self._rule_embeddings.size(0) > 0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        pad_mask = (x == self.pad_id)
        if pad_mask.all(dim=1).any():
            pad_mask[pad_mask.all(dim=1), 0] = False

        state_vec = self._encode_state(x, pad_mask)
        value = self.value_head(state_vec)

        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。")

        if self._rule_embeddings.device != x.device:
            self._rule_embeddings = self._rule_embeddings.to(x.device)

        rule_emb = self._rule_embeddings
        state_norm = F.normalize(state_vec, p=2, dim=-1)
        cosine_sim = torch.matmul(state_norm, rule_emb.T)

        temperature = torch.exp(self.log_temperature.clamp(max=math.log(100.0)))
        policy_logits = cosine_sim / temperature

        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, -1e9)

        return policy_logits, value

MathAlphaZeroNet = MathNet
