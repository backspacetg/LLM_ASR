# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
eps = 0.00001


def svd(mat: torch.Tensor, saved_rank: int):
    u, s, v = torch.svd(mat)
    vt = s.unsqueeze(1)*v.t()
    return u[:, :saved_rank], vt[:saved_rank, :], s


class SVDLinear(nn.Linear):
    """
    Fully Connected layer whose weights can be decomposed by SVD
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        out_features_split_num: int = 1
    ):
        super(SVDLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.svd_applied = False
        self.out_features_split_num = out_features_split_num
        self.out_features_split_dim = int(out_features/out_features_split_num)
        assert out_features % out_features_split_num == 0
    
    def apply_svd(self, keep_rate, log=True):
        keep_rank = int(min(self.weight.shape)*keep_rate)
        u_list, vt_list, s_list = self.split_layer_by_svd(keep_rank)
        if log:
            for s in s_list:
                logger.info(f"max/min/mean s {s.max().item()} {s.min().item()} {s.mean().item()}")
        self.weight_u_list = nn.ParameterList([nn.parameter.Parameter(u, requires_grad=True) for u in u_list])
        self.weight_vt_list = nn.ParameterList([nn.parameter.Parameter(vt, requires_grad=True) for vt in vt_list])
        self.weight = None
        return
    
    def split_layer_by_svd(self, keep_rank):
        u_list = []
        vt_list = []
        s_list = []
        for i in range(self.out_features_split_num):
            u, vt, s = svd(
                self.weight.data[i*self.out_features_split_dim:(i+1)*self.out_features_split_dim, :], 
                keep_rank
                )
            u_list.append(u)
            vt_list.append(vt)
            s_list.append(s)
        self.svd_applied = True
        return u_list, vt_list, s_list
    
    def forward(self, input: torch.tensor):
        if self.svd_applied:
            results = []
            for i, (weight_vt, weight_u) in enumerate(zip(self.weight_vt_list, self.weight_u_list)):
                after_vs = nn.functional.linear(input, weight_vt)
                result = nn.functional.linear(
                    after_vs, weight_u, 
                    self.bias[i*self.out_features_split_dim:(i+1)*self.out_features_split_dim]
                    )
                results.append(result)
            if self.out_features_split_num > 1:
                results = torch.cat(results, dim=-1)
                return results
            else:
                return results[0]
        else:
            return nn.functional.linear(input, self.weight, self.bias)

    

