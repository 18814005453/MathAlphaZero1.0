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
    - 严格静态规则缓存：规则数量一次性定死，禁止运行期扩容。
    - 策略头输出形状严格为 [Batch, Num_Rules]。
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

        # 规则缓存及映射表（初始为空）
        self._rule_embeddings = torch.empty(0, d_model)
        self._rule_to_action_idx: Dict[int, int] = {}
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _encode_state(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """状态塔编码，输出 [batch, d_model]"""
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        encoded = self.transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)

    def _encode_rules(self, rule_tokens: torch.Tensor) -> torch.Tensor:
        """规则塔编码，输出 [num_rules, d_model]"""
        mask = (rule_tokens == self.pad_id)
        # 避免全 padding 行导致 mask 全 True
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False

        emb = self.embedding(rule_tokens) * math.sqrt(self.d_model)
        emb = self.rule_pos_encoder(emb)
        encoded = self.rule_transformer_encoder(emb, src_key_padding_mask=mask)
        return self.pool(encoded, mask)  # [num_rules, d_model]

    def refresh_rule_cache(
            self,
            rule_texts: List[str],
            tokenizer_fn: Callable,
            action_ids: Optional[List[int]] = None
    ) -> Dict[int, int]:
        """
        刷新规则缓存。
        - 若 action_ids 为 None，自动生成为 0..N-1。
        - 若 action_ids 不为 None，长度必须等于 len(rule_texts)，否则抛出 ValueError。
        - 规则特征矩阵形状固定为 [num_rules, d_model]，一次性定死，永不动态扩容。
        - 返回 id_to_idx 映射。
        """
        device = next(self.parameters()).device
        if not rule_texts:
            self._rule_embeddings = torch.empty(0, self.d_model, device=device)
            self.id_to_idx = {}
            self.idx_to_id = {}
            self._rule_to_action_idx = {}
            return {}

        # 保底：自动生成 action_ids
        if action_ids is None:
            action_ids = list(range(len(rule_texts)))
        else:
            if len(action_ids) != len(rule_texts):
                raise ValueError(f"action_ids 长度 ({len(action_ids)}) 必须与规则文本数量 ({len(rule_texts)}) 一致")

        # 建立双向映射
        self.id_to_idx = {int(act_id): idx for idx, act_id in enumerate(action_ids)}
        self.idx_to_id = {idx: int(act_id) for idx, act_id in enumerate(action_ids)}
        self._rule_to_action_idx = self.id_to_idx  # 兼容旧接口

        # ========== 修改点：逐个规则分词并拼接 ==========
        token_list = []
        max_len = self.max_len   # 使用网络预设的最大长度
        for rule in rule_texts:
            # 每个规则独立调用 tokenizer_fn，假设返回形状 [seq_len] 或 [1, seq_len]
            tokens = tokenizer_fn(rule)   # 可能返回 Tensor 或 list，形状不定
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            else:
                tokens = tokens.to(device)

            # 确保为二维 [1, max_len]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)       # [1, L]
            # 截断或填充到 max_len
            if tokens.size(1) > max_len:
                tokens = tokens[:, :max_len]
            elif tokens.size(1) < max_len:
                pad = torch.full((1, max_len - tokens.size(1)), self.pad_id, dtype=tokens.dtype, device=device)
                tokens = torch.cat([tokens, pad], dim=1)
            token_list.append(tokens)

        # 沿第0维拼接 -> [num_rules, max_len]
        tokens = torch.cat(token_list, dim=0)
        # =============================================

        with torch.no_grad():
            rule_vecs = self._encode_rules(tokens)          # [N, d_model]
            rule_vecs = F.normalize(rule_vecs, p=2, dim=-1)  # 可选，但推荐保留

        # 一次性固定形状 [num_rules, d_model]
        self._rule_embeddings = rule_vecs.detach().clone()

        return self.id_to_idx

    def get_rule_index(self, action_id: int) -> int:
        """
        根据物理 Action ID 查找规则矩阵行索引。
        - 若 action_id 已注册，返回对应索引。
        - 若未注册，**绝不动态扩容**，仅返回安全索引（规则非空时返回最大合法索引，否则抛出异常）。
        - 同时打印警告信息，提示引擎应预先注册所有规则。
        """
        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空，无法获取规则索引。请先调用 refresh_rule_cache() 注入规则。")

        act_id_int = int(action_id)

        # 查找映射
        idx = self.id_to_idx.get(act_id_int)
        if idx is not None:
            return idx

        # 未注册：返回最大合法索引（最后一个规则的索引）
        max_idx = len(self._rule_embeddings) - 1
        print(f"[警告] 未知的 action_id={act_id_int}，未在规则缓存中注册。"
              f"返回安全索引 {max_idx}（对应规则索引 {max_idx}），"
              f"请确保在 forward 前通过 refresh_rule_cache() 注册所有可能出现的 action_id。")
        return max_idx

    def get_rule_embeddings(self) -> torch.Tensor:
        """返回规则特征矩阵 [num_rules, d_model]"""
        return self._rule_embeddings

    @property
    def rule_cache_valid(self) -> bool:
        return getattr(self, '_rule_embeddings', None) is not None and self._rule_embeddings.size(0) > 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        输入: x [batch, seq_len] (token ids)
        输出:
            policy_logits [batch, num_rules]  — 严格保持规则数量维度
            value [batch, 1]
        """
        mask = (x == self.pad_id)
        # 避免全 padding 行导致 mask 全 True 引发 NaN
        if mask.all(dim=1).any():
            mask[mask.all(dim=1), 0] = False

        # 状态编码
        state_vec = self._encode_state(x, mask)  # [batch, d_model]
        value = self.value_head(state_vec)       # [batch, 1]

        if not self.rule_cache_valid:
            raise RuntimeError("规则缓存为空。在 forward 之前，引擎必须调用 refresh_rule_cache() 注入规则。")

        # 设备对齐
        if self._rule_embeddings.device != x.device:
            self._rule_embeddings = self._rule_embeddings.to(x.device)

        rule_emb = self._rule_embeddings          # [num_rules, d_model]
        state_norm = F.normalize(state_vec, p=2, dim=-1)  # [batch, d_model]

        # ========== 关键：矩阵乘法，输出形状 [batch, num_rules] ==========
        cosine_sim = torch.matmul(state_norm, rule_emb.T)   # [batch, num_rules]

        # 温度缩放
        temperature = torch.exp(self.log_temperature.clamp(max=math.log(100.0)))
        policy_logits = cosine_sim / temperature            # [batch, num_rules]

        return policy_logits, value

    def freeze_state_encoder(self, freeze: bool = True):
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