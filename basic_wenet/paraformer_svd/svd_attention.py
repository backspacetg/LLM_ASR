from ..paraformer.attention import MultiHeadedAttentionSANM
from .svd_linear import SVDLinear

class MHA_SANM_SVD(MultiHeadedAttentionSANM):
    def __init__(
            self, n_head, 
            in_feat, n_feat, 
            dropout_rate, kernel_size, sanm_shfit=0
            ):
        super().__init__(
            n_head, in_feat, n_feat, dropout_rate, kernel_size, sanm_shfit)
        self.linear_out = SVDLinear(
            n_feat, n_feat, out_features_split_num=1)
        self.linear_q_k_v = SVDLinear(
            in_feat, n_feat*3, out_features_split_num=3)
