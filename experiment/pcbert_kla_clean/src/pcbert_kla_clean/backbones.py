from __future__ import annotations

import torch
from torch import nn


def is_t5_like_model(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    return (
        "prot_t5" in model_name_lower
        or "prott5" in model_name_lower
        or "ankh" in model_name_lower
    )


def infer_sequence_format(model_name: str) -> str:
    if model_name == "Rostlab/prot_bert":
        return "protbert_spaced"
    if is_t5_like_model(model_name):
        return "prott5_spaced"
    return "raw"


def infer_site_token_index(model_name: str) -> int:
    if is_t5_like_model(model_name):
        return 25
    return 26


def load_encoder(model_name: str, cache_dir: str | None):
    from transformers import AutoModel, BertModel, T5EncoderModel

    if model_name == "Rostlab/prot_bert":
        return BertModel.from_pretrained(model_name, cache_dir=cache_dir)
    if is_t5_like_model(model_name):
        return T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
    return AutoModel.from_pretrained(model_name, cache_dir=cache_dir)


def load_tokenizer(model_name: str, cache_dir: str | None):
    from transformers import AutoTokenizer, BertTokenizer

    if model_name == "Rostlab/prot_bert":
        return BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            do_lower_case=False,
        )

    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False,
    )


def truncate_encoder_layers(encoder: nn.Module, encoder_layers: int) -> None:
    if encoder_layers <= 0:
        raise ValueError("encoder_layers must be positive")

    transformer_encoder = getattr(encoder, "encoder", None)
    if transformer_encoder is None:
        raise ValueError("Expected an encoder-backed transformer model")

    if hasattr(transformer_encoder, "layer"):
        layers = transformer_encoder.layer
        if encoder_layers < len(layers):
            transformer_encoder.layer = nn.ModuleList(list(layers[:encoder_layers]))
        return

    if hasattr(transformer_encoder, "block"):
        layers = transformer_encoder.block
        if encoder_layers < len(layers):
            transformer_encoder.block = nn.ModuleList(list(layers[:encoder_layers]))
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
