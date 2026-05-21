# core/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Callable, Optional, Dict


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码"""

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
    """掩码均值池化（忽略 padding）"""

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        mask = ~src_key_padding_mask
        mask = mask.unsqueeze(-1).float()
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        valid_len = mask.sum(dim=1).clamp(min=1.0)
        return sum_x / valid_len


class MathNet(nn.Module):
    """
    AlphaZero 双塔解耦终极版：
    - 状态塔：沿用预训练权重，负责复杂题目特征提取。
    - 规则塔：拥有独立 Transformer 和位置编码，专注规则模板对齐。
    - 零开销设备缓存与严密越界防御机制。
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 3,  # 状态塔层数
            rule_num_layers: int = 2,  # 规则塔层数（轻量）
            max_len: int = 128,
            dropout: float = 0.1,
            temperature: float = 0.07,  # 固定温度（推荐值）
            learn_temperature: bool = False,
            num_actions: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = 0
        self.temperature = temperature

        # 共享词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)

        # 独立位置编码：隔离长短序列的分布干扰
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.rule_pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        self.pool = MaskedAvgPool()

        # 状态塔编码器（保留原有结构，无缝承接旧权重）
        state_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(state_encoder_layer, num_layers=num_layers)

        # 规则塔独立编码器（轻量，避免梯度干扰）
        rule_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.rule_transformer_encoder = nn.TransformerEncoder(rule_encoder_layer, num_layers=rule_num_layers)

        # 价值头 (无缝承接旧权重)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )

        # 温度系数处理
        if learn_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer('log_temperature', torch.tensor(math.log(temperature)))

        # 规则缓存及映射表
        self._rule_embeddings = torch.empty(0, d_model)
        self._rule_to_action_idx: Dict[int, int] = {}  # action_id -> rule_index

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_state(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """状态塔编码"""
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        encoded = self.transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)

    def _encode_rules(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        """规则塔编码（使用独立的位置编码和 Transformer）"""
        mask = (rule_tokens == self.pad_id)
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False

        emb = self.embedding(rule_tokens) * math.sqrt(self.d_model)
        emb = self.rule_pos_encoder(emb)  # 应用独立的规则位置编码
        encoded = self.rule_transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)

    def refresh_rule_cache(
            self,
            rule_texts: List[str],
            tokenizer_fn: Callable,
            action_ids: Optional[List[int]] = None
    ) -> Dict[int, int]:
        """刷新规则缓存，外部引擎负责在此处注入规则与动作的对应关系"""
        device = next(self.parameters()).device  # 获取模型当前首选设备
        if not rule_texts:
            self._rule_embeddings = torch.empty(0, self.d_model, device=device)
            self._rule_to_action_idx = {}
            return {}

        tokens = tokenizer_fn(rule_texts)
        if tokens.device != device:
            tokens = tokens.to(device)

        with torch.no_grad():
            rule_vecs = self._encode_rules(tokens)  # [N, d_model]
            rule_vecs = F.normalize(rule_vecs, p=2, dim=-1)  # 预归一化

        self._rule_embeddings = rule_vecs.detach().clone()

        # 建立严格的映射关系
        mapping = {}
        if action_ids is not None:
            if len(action_ids) != len(rule_texts):
                raise ValueError(f"action_ids 长度 ({len(action_ids)}) 与 rule_texts 长度 ({len(rule_texts)}) 不匹配！")
            for idx, aid in enumerate(action_ids):
                mapping[aid] = idx
        self._rule_to_action_idx = mapping

        return mapping

    def get_rule_index(self, action_id: int) -> int:
        """
        严密防御：根据 action_id 获取规则矩阵的行索引。
        拒绝静默回退，倒逼引擎侧保证动作映射的准确性。
        """
        if action_id not in self._rule_to_action_idx:
            raise ValueError(f"致命错误：动作 ID {action_id} 未在规则映射表中注册。请检查引擎是否漏传了 action_ids。")
        return self._rule_to_action_idx[action_id]

    def get_rule_embeddings(self) -> torch.Tensor:
        """
        [新增补全] 供引擎侧（engine.py）动态提取底层计算空间维度的只读缓存接口，
        用于通过严格断言校验并自适应调整搜索空间的动作剪枝。
        """
        return self._rule_embeddings

    @property
    def rule_cache_valid(self) -> bool:
        return getattr(self, '_rule_embeddings', None) is not None and self._rule_embeddings.size(0) > 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (x == self.pad_id)
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False

        # 状态向量抽取
        state_vec = self._encode_state(x, mask)  # [batch, d_model]
        value = self.value_head(state_vec)  # [batch, 1]

        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。在 forward 之前，引擎必须调用 refresh_rule_cache() 注入规则。")

        # 零开销设备对齐：仅在设备不一致时触发搬移，压榨 MCTS 吞吐率
        if self._rule_embeddings.device != x.device:
            self._rule_embeddings = self._rule_embeddings.to(x.device)

        rule_emb = self._rule_embeddings
        state_norm = F.normalize(state_vec, p=2, dim=-1)  # [batch, d_model]

        # 内积匹配
        cosine_sim = torch.matmul(state_norm, rule_emb.T)  # [batch, N]

        # 温度平滑
        temperature = torch.exp(self.log_temperature.clamp(max=math.log(100.0)))
        policy_logits = cosine_sim / temperature

        return policy_logits, value

    def freeze_state_encoder(self, freeze: bool = True):
        """保护预训练特征抽取器"""
        for param in self.transformer_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_pos_encoder(self, freeze: bool = True):
        for param in self.pos_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_rule_encoder(self, freeze: bool = True):
        for param in self.rule_transformer_encoder.parameters():
            param.requires_grad = not freeze

    def freeze_embedding(self, freeze: bool = True):
        for param in self.embedding.parameters():
            param.requires_grad = not freeze


# 兼容引擎调用
MathAlphaZeroNet = MathNet


def create_network(vocab_size, num_actions=None, **kwargs):
    default_params = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'rule_num_layers': 2,
        'max_len': 128,
        'dropout': 0.1,
        'temperature': 0.07,
        'learn_temperature': False
    }
    default_params.update(kwargs)
    default_params.pop('num_actions', None)
    return MathNet(vocab_size, **default_params)