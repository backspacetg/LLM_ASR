import torch
from ..paraformer.layers import SanmEncoder, AliParaformerEncoderLayer
from .svd_attention import MHA_SANM_SVD
from .svd_feed_forward import SVDPositionwiseFeedForward


class SVDSanmEncoder(SanmEncoder):
    def __init__(
            self, 
            input_size: int, 
            output_size: int = 256, 
            attention_heads: int = 4, 
            linear_units: int = 2048, 
            num_blocks: int = 6, 
            dropout_rate: float = 0.1, 
            positional_dropout_rate: float = 0.1, 
            attention_dropout_rate: float = 0, 
            input_layer: str = "conv2d", 
            pos_enc_layer_type: str = "abs_pos", 
            normalize_before: bool = True, 
            static_chunk_size: int = 0, 
            use_dynamic_chunk: bool = False, 
            global_cmvn: torch.nn.Module = None, 
            use_dynamic_left_chunk: bool = False, 
            kernel_size: int = 11, 
            sanm_shfit: int = 0, 
            gradient_checkpointing: bool = False
        ):
        super().__init__(
            input_size, 
            output_size, 
            attention_heads, 
            linear_units, 
            num_blocks, 
            dropout_rate, 
            positional_dropout_rate, 
            attention_dropout_rate, 
            input_layer, 
            pos_enc_layer_type, 
            normalize_before, 
            static_chunk_size, 
            use_dynamic_chunk, 
            global_cmvn, 
            use_dynamic_left_chunk, 
            kernel_size, 
            sanm_shfit, 
            gradient_checkpointing)

        self.encoders = torch.nn.ModuleList([
            AliParaformerEncoderLayer(
                output_size,
                MHA_SANM_SVD(
                    n_head=attention_heads, 
                    in_feat=output_size, 
                    n_feat=output_size, 
                    dropout_rate=attention_dropout_rate, 
                    kernel_size=kernel_size, 
                    sanm_shfit=sanm_shfit),
                SVDPositionwiseFeedForward(
                    output_size,
                    linear_units,
                    dropout_rate,
                ),
                dropout_rate,
                normalize_before,
                in_size=output_size) for _ in range(num_blocks - 1)
        ])