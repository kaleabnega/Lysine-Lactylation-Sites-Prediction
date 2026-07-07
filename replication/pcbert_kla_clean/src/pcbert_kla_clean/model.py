from __future__ import annotations

import torch
from torch import nn


def is_t5_like_model(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    return "prot_t5" in model_name_lower or "prott5" in model_name_lower


def load_encoder(model_name: str, cache_dir: str | None):
    from transformers import AutoModel, BertModel, T5EncoderModel

    if model_name == "Rostlab/prot_bert":
        return BertModel.from_pretrained(model_name, cache_dir=cache_dir)
    if is_t5_like_model(model_name):
        return T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    return AutoModel.from_pretrained(model_name, cache_dir=cache_dir)


def truncate_encoder_layers(encoder: nn.Module, encoder_layers: int) -> None:
    if encoder_layers <= 0:
        raise ValueError("encoder_layers must be positive")

    bert_encoder = getattr(encoder, "encoder", None)
    if bert_encoder is None:
        raise ValueError("Expected an encoder-backed transformer model")

    if hasattr(bert_encoder, "layer"):
        layers = bert_encoder.layer
        if encoder_layers < len(layers):
            bert_encoder.layer = nn.ModuleList(list(layers[:encoder_layers]))
        return

    if hasattr(bert_encoder, "block"):
        layers = bert_encoder.block
        if encoder_layers < len(layers):
            bert_encoder.block = nn.ModuleList(list(layers[:encoder_layers]))
        return

    raise ValueError("Expected encoder.layer or encoder.block")


def set_encoder_trainable(encoder: nn.Module, trainable: bool) -> None:
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable


def encode_with_optional_freeze(
    encoder: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    freeze_encoder: bool,
):
    if not freeze_encoder:
        return encoder(**model_inputs)

    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        outputs = encoder(**model_inputs)
    if was_training:
        encoder.train()
    return outputs


def encoder_hidden_size(encoder: nn.Module) -> int:
    hidden_size = getattr(encoder.config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    d_model = getattr(encoder.config, "d_model", None)
    if d_model is not None:
        return int(d_model)

    raise ValueError("Encoder config must expose hidden_size or d_model")


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
        freeze_encoder: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = load_encoder(model_name, cache_dir)
        truncate_encoder_layers(self.encoder, encoder_layers)
        set_encoder_trainable(self.encoder, trainable=not freeze_encoder)

        hidden_size = encoder_hidden_size(self.encoder)
        self.fc1 = nn.Linear(hidden_size + feature_dim, 32)
        self.attention = FeatureAttention(32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)

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

        outputs = encode_with_optional_freeze(
            self.encoder,
            model_inputs,
            self.freeze_encoder,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :].to(features.dtype)
        x = torch.cat((cls_embedding, features), dim=1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.attention(x)
        x = self.dropout2(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        return torch.sigmoid(self.fc3(x)).squeeze(-1)


class SiteAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_dim: int = 256,
        site_token_index: int = 26,
        has_start_token: bool = True,
    ) -> None:
        super().__init__()
        self.site_token_index = site_token_index
        self.has_start_token = has_start_token
        self.token_projection = nn.Linear(hidden_size, attention_dim)
        self.site_projection = nn.Linear(hidden_size, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = token_embeddings.shape[1]
        site_index = min(self.site_token_index, sequence_length - 1)
        site_embedding = token_embeddings[:, site_index, :]

        logits = self.score(
            torch.tanh(
                self.token_projection(token_embeddings)
                + self.site_projection(site_embedding).unsqueeze(1)
            )
        ).squeeze(-1)

        residue_mask = attention_mask.bool().clone()
        if self.has_start_token:
            residue_mask[:, 0] = False
        sep_indices = attention_mask.long().sum(dim=1) - 1
        residue_mask[
            torch.arange(residue_mask.shape[0], device=residue_mask.device),
            sep_indices,
        ] = False

        logits = logits.masked_fill(~residue_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return torch.sum(weights * token_embeddings, dim=1)


class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence_embedding: torch.Tensor,
        feature_embedding: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate(torch.cat((sequence_embedding, feature_embedding), dim=1))
        return gate * sequence_embedding + (1.0 - gate) * feature_embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TokenGatedPCBertKla(nn.Module):
    """PCBert-Kla variant with site-aware token pooling and gated feature fusion."""

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        feature_dim: int = 27,
        encoder_layers: int = 4,
        fusion_dim: int = 256,
        attention_dim: int = 256,
        dropout: float = 0.2,
        site_token_index: int = 26,
        freeze_encoder: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = load_encoder(model_name, cache_dir)
        truncate_encoder_layers(self.encoder, encoder_layers)
        set_encoder_trainable(self.encoder, trainable=not freeze_encoder)

        hidden_size = encoder_hidden_size(self.encoder)
        self.site_attention = SiteAttentionPooling(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            site_token_index=site_token_index,
            has_start_token=not is_t5_like_model(model_name),
        )
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = GatedFusion(fusion_dim)
        self.classifier = nn.Sequential(
            ResidualBlock(fusion_dim, dropout=dropout),
            ResidualBlock(fusion_dim, dropout=dropout),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

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

        outputs = encode_with_optional_freeze(
            self.encoder,
            model_inputs,
            self.freeze_encoder,
        )
        token_embeddings = outputs.last_hidden_state.to(features.dtype)
        sequence_embedding = self.site_attention(token_embeddings, attention_mask)
        sequence_embedding = self.sequence_projection(sequence_embedding)
        feature_embedding = self.feature_projection(features)
        fused = self.fusion(sequence_embedding, feature_embedding)
        return torch.sigmoid(self.classifier(fused)).squeeze(-1)


class HybridTokenGatedPCBertKla(nn.Module):
    """PCBert-Kla variant that preserves CLS context and adds site-aware pooling."""

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        feature_dim: int = 27,
        encoder_layers: int = 4,
        fusion_dim: int = 256,
        attention_dim: int = 256,
        dropout: float = 0.2,
        site_token_index: int = 26,
        freeze_encoder: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = load_encoder(model_name, cache_dir)
        truncate_encoder_layers(self.encoder, encoder_layers)
        set_encoder_trainable(self.encoder, trainable=not freeze_encoder)

        hidden_size = encoder_hidden_size(self.encoder)
        self.site_attention = SiteAttentionPooling(
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            site_token_index=site_token_index,
            has_start_token=not is_t5_like_model(model_name),
        )
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = GatedFusion(fusion_dim)
        self.classifier = nn.Sequential(
            ResidualBlock(fusion_dim, dropout=dropout),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

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

        outputs = encode_with_optional_freeze(
            self.encoder,
            model_inputs,
            self.freeze_encoder,
        )
        token_embeddings = outputs.last_hidden_state.to(features.dtype)
        cls_embedding = token_embeddings[:, 0, :]
        site_embedding = self.site_attention(token_embeddings, attention_mask)
        sequence_embedding = self.sequence_projection(
            torch.cat((cls_embedding, site_embedding), dim=1)
        )
        feature_embedding = self.feature_projection(features)
        fused = self.fusion(sequence_embedding, feature_embedding)
        return torch.sigmoid(self.classifier(fused)).squeeze(-1)
