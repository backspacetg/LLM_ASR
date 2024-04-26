#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm
from ..paraformer.embedding import ParaformerPositinoalEncoding
from ..transformer.norm import RMSNorm
from ..transformer.positionwise_feed_forward import (
    GatedVariantsMLP, MoEFFNLayer, PositionwiseFeedForward)

from ..transformer.swish import Swish
from ..transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    StackNFramesSubsampling,
)

from ..transformer.embedding import (PositionalEncoding,
                                         RelPositionalEncoding,
                                         WhisperPositionalEncoding,
                                         LearnablePositionalEncoding,
                                         NoPositionalEncoding)
from ..transformer.attention import (MultiHeadedAttention,
                                         MultiHeadedCrossAttention,
                                         RelPositionMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}

WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity,
    'stack_n_frames': StackNFramesSubsampling,
}

WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    "abs_pos_paraformer": ParaformerPositinoalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
    "crossattn": MultiHeadedCrossAttention,
    'shaw_rel_selfattn': ShawRelPositionMultiHeadedAttention
}

WENET_MLP_CLASSES = {
    'position_wise_feed_forward': PositionwiseFeedForward,
    'moe': MoEFFNLayer,
    'gated': GatedVariantsMLP
}

WENET_NORM_CLASSES = {
    'layer_norm': LayerNorm,
    'batch_norm': BatchNorm1d,
    'rms_norm': RMSNorm
}
