# core/network.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码，用于给序列中的每个位置注入位置信息。
    编码维度 d_model 必须与词嵌入维度一致。
    """

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建固定的位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MaskedAvgPool(nn.Module):
    """
    掩码均值池化：忽略填充位置（pad_id=0）对序列做平均。
    输入: x [batch_size, seq_len, d_model],  src_key_padding_mask [batch_size, seq_len] (True 表示忽略位置)
    输出: [batch_size, d_model]
    """

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        # src_key_padding_mask: True 表示该位置是 padding，应当忽略
        # 将 padding 位置的贡献置零
        mask = ~src_key_padding_mask  # True 表示有效位置
        mask = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)  # [batch, d_model]

        # 有效 token 数量（避免除零）
        valid_len = mask.sum(dim=1)  # [batch, 1]
        valid_len = valid_len.clamp(min=1.0)
        return sum_x / valid_len


class MathNet(nn.Module):
    """
    用于符号积分的 AlphaZero 风格神经网络（原名 MathAlphaZeroNet，已完美对接 MCTS 引擎类型注解）。
    输入: 经过预处理的 token 序列（已经 pad 到固定长度，PAD_ID = 0）
    输出: policy_logits (batch_size, num_actions)
          value (batch_size, 1)，范围 [-1, 1]
    """

    def __init__(
            self,
            vocab_size: int,
            num_actions: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 3,
            max_len: int = 128,
            dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = 0  # 固定 padding id 为 0

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 使用 batch first 风格，输入形状 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 池化层
        self.pool = MaskedAvgPool()

        # 策略头 (Policy Head)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_actions)
        )

        # 价值头 (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: LongTensor, shape (batch_size, seq_len). 每个元素是 token id，0 为填充符。
        返回:
            policy_logits: (batch_size, num_actions)
            value: (batch_size, 1)
        """
        # 生成 padding mask: True 表示该位置需要被忽略（是填充符）
        src_key_padding_mask = (x == self.pad_id)  # [batch, seq_len]

        # 鲁棒性防御：防止全 Padding 序列造成 Transformer 产生全 NaN 输出
        # 如果某一行全为 True，强行将第一个 Token 标记为有效（防止自注意力机制分母变 0）
        all_padded = src_key_padding_mask.all(dim=1)
        if all_padded.any():
            src_key_padding_mask[all_padded, 0] = False

        # 嵌入 + 位置编码
        # embedding 输出 [batch, seq, d_model]
        emb = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        emb = self.pos_encoder(emb)

        # Transformer 编码器
        # src_key_padding_mask 告诉编码器哪些位置是 pad，自注意力时不参与
        encoded = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)

        # 全局池化，得到整个表达式的表示
        global_repr = self.pool(encoded, src_key_padding_mask)  # [batch, d_model]

        # 策略头与价值头
        policy_logits = self.policy_head(global_repr)  # [batch, num_actions]
        value = self.value_head(global_repr)  # [batch, 1]

        return policy_logits, value


# 别名映射，完美兼容你之前编写的 MCTS 引擎里的类型注解 `from core.network import MathNet`
MathAlphaZeroNet = MathNet


def create_network(vocab_size, num_actions, **kwargs):
    """
    便捷函数，根据配置创建网络实例。
    """
    default_params = {
        'd_model': 128,
        'nhead': 4,
        'num_layers': 3,
        'max_len': 128,
        'dropout': 0.1
    }
    default_params.update(kwargs)
    return MathNet(vocab_size, num_actions, **default_params)