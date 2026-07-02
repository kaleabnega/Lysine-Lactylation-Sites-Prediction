from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, BertModel


class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(input_dim, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.weight(x)) * x


class PCBertKla(nn.Module):
    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        feature_dim: int = 27,
        encoder_layers: int = 4,
        dropout1: float = 0.1,
        dropout2: float = 0.3,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = self._load_encoder(model_name, cache_dir)
        self._truncate_encoder_layers(encoder_layers)

        hidden_size = int(self.encoder.config.hidden_size)
        self.fc1 = nn.Linear(hidden_size + feature_dim, 32)
        self.attention = FeatureAttention(32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)

    @staticmethod
    def _load_encoder(model_name: str, cache_dir: str | None):
        if model_name == "Rostlab/prot_bert":
            return BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        return AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

    def _truncate_encoder_layers(self, encoder_layers: int) -> None:
        if encoder_layers <= 0:
            raise ValueError("encoder_layers must be positive")

        bert_encoder = getattr(self.encoder, "encoder", None)
        if bert_encoder is None or not hasattr(bert_encoder, "layer"):
            raise ValueError("Expected a BERT-like model with encoder.layer")

        layers = bert_encoder.layer
        if encoder_layers < len(layers):
            bert_encoder.layer = nn.ModuleList(list(layers[:encoder_layers]))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = torch.cat((cls_embedding, features), dim=1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.attention(x)
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.fc3(x)).squeeze(-1)
