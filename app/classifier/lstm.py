import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from jaxtyping import Float, Int
from typing import Literal
from .base import Classifier


class LSTMClassifier(Classifier):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        reduction: Literal["mean", "first", "last"] = "last",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 计算输出维度（双向LSTM时是2倍）
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.reduction = reduction
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes),
        )

    def forward(
        self,
        x: Int[torch.Tensor, "... seq_len"],
        len: Int[torch.Tensor, "..."] | None = None,
    ) -> Float[torch.Tensor, "... num_classes"]:
        # 嵌入层
        x = self.embedding(x)
        
        # 处理序列长度
        if len is not None:
            # 使用 pack_padded_sequence 提高效率
            packed_x = pack_padded_sequence(x, len.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out_packed, (hidden, cell) = self.lstm(packed_x)
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # 根据 reduction 策略选择输出
        if self.reduction == "last":
            if len is not None:
                # 获取每个序列的最后一个有效位置
                batch_size = lstm_out.size(0)
                idx = (len - 1).unsqueeze(1).unsqueeze(2).expand(
                    batch_size, 1, lstm_out.size(2)
                )
                x = lstm_out.gather(1, idx).squeeze(1)
            else:
                x = lstm_out[:, -1, :]
        elif self.reduction == "first":
            x = lstm_out[:, 0, :]
        elif self.reduction == "mean":
            if len is not None:
                # 使用掩码计算平均值
                mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0) < len.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                x = (lstm_out * mask).sum(dim=1) / len.unsqueeze(-1)
            else:
                x = lstm_out.mean(dim=1)
        
        # 分类头
        x = self.classifier(x)
        return x

