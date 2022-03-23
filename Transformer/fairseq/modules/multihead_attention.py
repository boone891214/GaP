# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from typing import Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.variable  import Variable
import strided_batched_gemm
import numpy as np

from fairseq import utils

global_cnt = 0

class QueryLinear(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, weights_q, scale) :
        s = Variable(torch.tensor([scale]))
        ctx.save_for_backward(input, weights_q, s)
        q = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_q, beta=0.0, alpha=s[0])
        q=q.view(input.size(0), input.size(1), input.size(2))
        return q.detach()

    @staticmethod
    def backward(ctx, q_grad) :
        input,weights_q,s = ctx.saved_tensors
        input = input.view(input.size(0)*input.size(1), input.size(2)).transpose(0,1)
        q = torch.addmm(q_grad.view(q_grad.size(0)*q_grad.size(1), q_grad.size(2)), q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)), weights_q.transpose(0,1), beta=0.0, alpha=s[0])
        q=q.view(q_grad.size(0), q_grad.size(1), q_grad.size(2))
        q_grad = q_grad.view(q_grad.size(0)*q_grad.size(1), q_grad.size(2))
        weights_q_grad = torch.addmm(weights_q, input, q_grad, beta=0.0, alpha=s[0])
        return q, weights_q_grad, None


class KeyValueLinears(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, weights_k, weights_v) :
        ctx.save_for_backward(input, weights_k, weights_v)
        k = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_k, beta=0.0, alpha=1.0)
        k=k.view(input.size(0), input.size(1), input.size(2))
        v = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_v, beta=0.0, alpha=1.0)
        v=v.view(input.size(0), input.size(1), input.size(2))
        return k.detach(),v.detach()

    @staticmethod
    def backward(ctx, k_grad, v_grad) :
        input,weights_k, weights_v = ctx.saved_tensors
        input = input.view(input.size(0)*input.size(1), input.size(2)).transpose(0,1)
        k = torch.addmm(k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)), k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)), weights_k.transpose(0,1), beta=0.0)
        k_grad = k_grad.view(k_grad.size(0)*k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)), weights_v.transpose(0,1), beta=1.0)
        v=v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0)*v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_k_grad, weights_v_grad


class SelfAttentionLinears(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, weights_q, weights_k, weights_v, scale) :
        s = Variable(torch.tensor([scale]))
        ctx.save_for_backward(input, weights_q, weights_k, weights_v, s)
        #print("input, w_q,k,v shape before,{},{},{},{}".format(input.shape, weights_q.shape, weights_k.shape,weights_v.shape))
        #reshaped_input = input.view(input.size(0)*input.size(1), input.size(2))
        #print("reshaped input:{}".format(reshaped_input.shape))
        q = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_q, beta=0.0, alpha=s[0])
        q=q.view(input.size(0), input.size(1), input.size(2))
        k = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_k, beta=0.0, alpha=1.0)
        k=k.view(input.size(0), input.size(1), input.size(2))
        v = torch.addmm(input.view(input.size(0)*input.size(1), input.size(2)), input.view(input.size(0) * input.size(1), input.size(2)), weights_v, beta=0.0, alpha=1.0)
        v=v.view(input.size(0), input.size(1), input.size(2))
        #print("input, q,k,v shape after,{},{},{},{}".format(input.shape, q.shape, k.shape, v.shape))
        #exit()
        return q.detach(),k.detach(),v.detach()

    @staticmethod
    def backward(ctx, q_grad, k_grad, v_grad) :
        input,weights_q,weights_k, weights_v,s = ctx.saved_tensors
        input = input.view(input.size(0)*input.size(1), input.size(2)).transpose(0,1)
        q = torch.addmm(q_grad.view(q_grad.size(0)*q_grad.size(1), q_grad.size(2)), q_grad.view(q_grad.size(0) * q_grad.size(1), q_grad.size(2)), weights_q.transpose(0,1), beta=0.0, alpha=s[0])
        q_grad = q_grad.view(q_grad.size(0)*q_grad.size(1), q_grad.size(2))
        weights_q_grad = torch.addmm(weights_q, input, q_grad, beta=0.0, alpha=s[0])
        k = q.addmm_(k_grad.view(k_grad.size(0) * k_grad.size(1), k_grad.size(2)), weights_k.transpose(0,1), beta=1.0)
        k_grad = k_grad.view(k_grad.size(0)*k_grad.size(1), k_grad.size(2))
        weights_k_grad = torch.mm(input, k_grad)
        v = k.addmm_(v_grad.view(v_grad.size(0) * v_grad.size(1), v_grad.size(2)), weights_v.transpose(0,1), beta=1.0)
        v=v.view(v_grad.size(0), v_grad.size(1), v_grad.size(2))
        v_grad = v_grad.view(v_grad.size(0)*v_grad.size(1), v_grad.size(2))
        weights_v_grad = torch.mm(input, v_grad)
        return v, weights_q_grad, weights_k_grad, weights_v_grad, None


class StridedBmm1Func(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input1, input2) :
        ctx.save_for_backward(input1, input2)
        output = torch.empty((input1.size(0),input1.size(1),input2.size(2)), dtype=input1.dtype, device=torch.device('cuda'))
        if (input1.dtype == torch.float16) and (input2.dtype == torch.float16) :
            output = strided_batched_gemm.strided_batched_gemm(0.0, output, 1.0, input1, input2)
        else :
            output = torch.bmm(input1, input2, out=output)
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output) :
        input1,input2 = ctx.saved_tensors
        grad_input1 = torch.empty((input1.size(1), input2.size(0), input1.size(2)), dtype=input1.dtype, device=torch.device('cuda')).transpose(1,0)
        grad_input2 = torch.empty((input2.size(2), input2.size(0), input2.size(1)), dtype=input2.dtype, device=torch.device('cuda')).transpose(1,0)
        if (grad_output.dtype == torch.float16) and (input1.dtype == torch.float16) and (input2.dtype == torch.float16) :
            grad_input1 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input1, 1.0, grad_output, input2.transpose(1,2))
            grad_input2 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input2, 1.0, grad_output.transpose(1,2), input1)
            grad_input2 = grad_input2.transpose(1,2)
        else :
            grad_input1 = torch.bmm(grad_output, input2.transpose(1,2), out=grad_input1)
            grad_input2 = torch.bmm(grad_output.transpose(1,2), input1, out=grad_input2).transpose(1,2)
        return grad_input1,grad_input2


class StridedBmm2Func(torch.autograd.Function) :
     @staticmethod
     def forward(ctx, input1, input2) :
         ctx.save_for_backward(input1, input2)
         output = torch.empty((input1.size(1), input1.size(0), input2.size(2)), dtype=input1.dtype, device=torch.device('cuda')).transpose(1,0)
         if (input1.dtype == torch.float16) and (input2.dtype == torch.float16) :
             output = strided_batched_gemm.strided_batched_gemm(0.0, output, 1.0, input1, input2)
         else:
             output = torch.bmm(input1, input2, out=output)
         return output.detach()

     @staticmethod
     def backward(ctx, grad_output) :
         input1,input2 = ctx.saved_tensors
         grad_input2 = torch.empty((input2.size(1), input2.size(0), input2.size(2)), dtype=input2.dtype, device=torch.device('cuda')).transpose(1,0)
         grad_input1 = torch.empty((input1.size(0), input1.size(1), input1.size(2)), dtype=input2.dtype, device=torch.device('cuda'))
         if (grad_output.dtype == torch.float16) and (input1.dtype == torch.float16) and (input2.dtype == torch.float16) :
             grad_input1 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input1, 1.0, grad_output, input2.transpose(1,2))
             grad_input2 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input2, 1.0, input1.transpose(1,2), grad_output)
         else :
             grad_input1 = torch.bmm(grad_output, input2.transpose(1,2))
             grad_input2 = torch.bmm(input1.transpose(1,2), grad_output, out=grad_input2)
         return grad_input1,grad_input2


def query_linear(input: Tensor, weights_q: Tensor, scale: float):
    if not torch.jit.is_scripting():
        return QueryLinear.apply(input, weights_q, scale)
    else:
        q = scale * torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_q)
        q = q.view(input.shape)
        return q

def key_value_linears(input: Tensor, weights_k: Tensor, weights_v: Tensor):
    if not torch.jit.is_scripting():
        return KeyValueLinears.apply(input, weights_k, weights_v)
    else:
        k = torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_k)
        k = k.view(input.shape)
        v = torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_v)
        v = v.view(input.shape)
        return k, v

def self_attn_linears(input: Tensor, weights_q: Tensor, weights_k: Tensor, weights_v: Tensor, scale:float):

    if not torch.jit.is_scripting():
        return SelfAttentionLinears.apply(input, weights_q, weights_k, weights_v, scale)
    else:

        q = scale * torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_q)
        q = q.view(input.shape)
        k = torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_k)
        k = k.view(input.shape)
        v = torch.einsum('ij,jk->ik', input.view(input.size(0)*input.size(1), -1), weights_v)
        v = v.view(input.shape)
        return q, k, v

def strided_bmm1(input1: Tensor, input2: Tensor):
    if not torch.jit.is_scripting():
        return StridedBmm1Func.apply(input1, input2)
    else:
        return  torch.einsum('ijk,ikn->ijn', input1, input2)

def strided_bmm2(input1: Tensor, input2: Tensor):
    if not torch.jit.is_scripting():
        return StridedBmm2Func.apply(input1, input2)
    else:
        return torch.einsum('ijk,ikn->ijn', input1, input2)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, args=None, test_only=False, en_de=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = torch.empty(0)
#        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        self.in_proj_weight_q = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_k = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.in_proj_weight_v = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
#            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
            self.in_proj_bias_q = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_k = Parameter(torch.Tensor(embed_dim))
            self.in_proj_bias_v = Parameter(torch.Tensor(embed_dim))
        else:
#            self.register_parameter('in_proj_bias', None)
            self.register_parameter('in_proj_bias_k', None)
            self.register_parameter('in_proj_bias_q', None)
            self.register_parameter('in_proj_bias_v', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_id = str(id(self))

        self.args = args

        self.debug = False #args.mhatt_debug
        torch.manual_seed(0)

        def mask_gen(pattern='random', nr=64, nc=64):
            mask = np.zeros([self.num_heads, 64, 64])
            if pattern == 'random':
                mask = (np.random.random(self.base_att_mask.shape) > 0.5 + 0.0) * (-1000)
            elif pattern == 'diagonal':
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        for k in range(mask.shape[2]):
                            if abs(j-k) > nr/2:
                                mask[i][j][k] = -1000
            return mask

        self.en_de = None
        if self.en_de is not None:
            self.apply_mask_en_only = True
            self.base_att_mask = Parameter(
                    torch.Tensor(self.num_heads, 64, 64),
                    #torch.Tensor(8, 8, 8),
                    requires_grad = False)
            if not test_only:
                np_base_att_mask = mask_gen('diagonal')
                self.base_att_mask.data = torch.FloatTensor(np_base_att_mask)
        else:
            self.apply_mask_en_only = False

        self.apply_mask = False #self.args.apply_att_mask and self.apply_mask_en_only



        self.att_mask = torch.FloatTensor(1,1,1)
        self.max_dim = [0,0,0]
        #print(debug)
        #input("?")
        self.reset_parameters()

    def reset_parameters(self):
#        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.in_proj_weight_q)
        nn.init.xavier_uniform_(self.in_proj_weight_k)
        nn.init.xavier_uniform_(self.in_proj_weight_v)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias_k is not None:
#            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.in_proj_bias_q, 0.)
            nn.init.constant_(self.in_proj_bias_k, 0.)
            nn.init.constant_(self.in_proj_bias_v, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask_future_timesteps: bool,
                key_padding_mask: Optional[Tensor],
                incremental_state: Optional[Dict[str, Dict[str, Tensor]]],
                need_weights: bool,
                static_kv: bool):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        if torch.jit.is_scripting():
            kv_same = torch.equal(key, value)
            qkv_same = torch.equal(query, value) and kv_same
        else:
            qkv_same, kv_same = self._fast_same_check(query, key, value)


        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        k = v = query.new_empty(0)
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        DEBUG = self.debug

        if qkv_same:
            # self-attention

            if DEBUG:
                print("qkv_same: query, w_q, w_k,w_v shapes:{},{},{},{}".format(query.shape, self.in_proj_weight_q.shape, self.in_proj_weight_k.shape, self.in_proj_weight_v.shape))
            q, k, v = self_attn_linears(query, self.in_proj_weight_q, self.in_proj_weight_k, self.in_proj_weight_v, self.scaling)
            if DEBUG:
                print("qkv_same: q,k,v shapes:{},{},{}".format(q.shape, k.shape, v.shape))
        elif kv_same:
            # encoder-decoder attention

            q = query_linear(query,self.in_proj_weight_q, self.scaling)
            if DEBUG:
                print("kv_same: query, w_q, q: {},{},{}".format(query.shape, self.in_proj_weight_q.shape, q.shape))
            if not(saved_state is not None and 'prev_key' in saved_state and static_kv):
                k, v = key_value_linears(key ,self.in_proj_weight_k, self.in_proj_weight_v)
            if DEBUG:
                print("kv_same: key, w_k, w_v, k, v shapes:{},{},{}".format(key.shape, self.in_proj_weight_k.shape, self.in_proj_weight_v.shape, k.shape, v.shape))

        else:
            q = torch.addmm(query.view(query.size(0)*query.size(1), query.size(2)), query.view(query.size(0) * query.size(1), query.size(2)), self.in_proj_weight_q, beta=0.0, alpha=self.scaling)
            if not(saved_state is not None and 'prev_key' in saved_state and static_kv):
                k = F.linear(key, self.in_proj_weight_k, self.in_proj_bias_k)
                v = F.linear(value, self.in_proj_weight_v, self.in_proj_bias_v)
            if DEBUG:
                print("all not the same: query, w_a, q:{},{},{}".format(query.shape, self.in_proj_weight_q.shape, q.shape))
                print("all not the same: key, w_k, k :{},{},{}".format(key.shape, self.in_proj_weight_k.shape, k.shape))
                print("all not the same: value, w_v, v:{},{},{}".format(value.shape, self.in_proj_weight_v.shape, v.shape))

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)

        if DEBUG:
            print("attention q*k shapes:{},{}".format(q.shape, k.shape))
        attn_weights = strided_bmm1(q, k.transpose(1, 2))
        if DEBUG:
            print("attn_weights shapes:{}".format(attn_weights.shape))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        att_shape = attn_weights.shape

        if self.apply_mask:
            if self.att_mask.shape[0] != attn_weights.shape[0] or \
                self.att_mask.shape[1] != attn_weights.shape[1] or  \
                self.att_mask.shape[2] != attn_weights.shape[2]:
                if DEBUG:
                    print("Setting masks...")
                repeat = [int(attn_weights.shape[0]/self.base_att_mask.shape[0])+1,
                          int(attn_weights.shape[1]/self.base_att_mask.shape[1])+1,
                          int(attn_weights.shape[2]/self.base_att_mask.shape[2])+1]
                #print(repeat)
                #print("Repeat:", repeat)
                #repeat = np.array(repeat)
                #del self.super_att_mask
                #torch.cuda.empty_cache()
                self.super_att_mask = self.base_att_mask.repeat(repeat[0], repeat[1], repeat[2])#.cuda().half()
                #self.att_mask = torch.ones(attn_weights.shape[0],attn_weights.shape[1],attn_weights.shape[2]).cuda().half()
                self.att_mask = self.super_att_mask[:att_shape[0],:att_shape[1],:att_shape[2]]
                if DEBUG:
                    print("ATT mask:", self.att_mask.shape)
            #self.att_mask = self.base_att_mask[:att_shape[0],:att_shape[1],:att_shape[2]].cuda()

        #for i in [0,1,2]:
        #    self.max_dim[i] = max(self.max_dim[i], att_shape[i])
        #print(self.max_dim)
        if DEBUG:
            #print("ATT mask:", att_mask.shape)
            #print(self.att_mask[0,:8,:8].detach().cpu().numpy())
            pass

        if DEBUG:
            aw = attn_weights[1,2,:].detach().cpu().numpy()
            print("Before softmax attn_weights values is:{}".format(aw))

        if self.apply_mask:
            attn_weights.add_(self.att_mask)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if False: #self.args.apply_dynamic_att_mask:

            sp = 0.5
            each_att_torch = attn_weights.reshape((-1,attn_weights.shape[-1]))
            #print("each att:", each_att_torch.shape)
            topk_torch = torch.topk(each_att_torch, max(1,int(each_att_torch.shape[1]*(1-sp))), dim=1)
            #print("top k:", topk_torch[0].shape)

            thr_torch = topk_torch[0][:,-1].reshape((each_att_torch.shape[0],1))

            thr_torch = torch.repeat_interleave(thr_torch, each_att_torch.shape[-1], dim=1)
            thr_torch = thr_torch.reshape(attn_weights.shape)
            mask  = (attn_weights  >= thr_torch) + 0.0
            attn_weights = attn_weights * mask

            #[bs, h, seq_len, seq_len] = [4,16,384,384]

            import time

        if DEBUG:
            aw = attn_weights[1,2,:].detach().cpu().numpy()
            print("attn_weights values is:{}".format(aw))
            print("Sum: ", np.sum(aw))
            print("Sorted: ", np.sort(aw))

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        if DEBUG:
            print("attention att*v shapes:{},{}".format(attn_weights.shape, v.shape))
        attn = strided_bmm2(attn_weights, v)
        if DEBUG:
            print("attn shapes:{}".format(attn.shape))

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        if DEBUG:
            print("before projection shapes:{}".format(attn.shape))
        attn = self.out_proj(attn)
        if DEBUG:
            print("after projection shapes:{}".format(attn.shape))
        global global_cnt
        global_cnt+=1
        if DEBUG:
            print("Multiheaded attention called:{}".format(global_cnt))
            import time
            time.sleep(5)
        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = attn_weights.new_empty(0) #Can't set to None because jit script reasons

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask.size(0) == 0:
            #TODO: try torch.new_full instead
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new_empty(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str,Tensor]]]):
        if incremental_state is None or self.cache_id not in incremental_state:
            return {}
        return incremental_state[self.cache_id]

    def _set_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str,Tensor]]], buffer: Dict[str,Tensor]):
        if incremental_state is not None:
            incremental_state[self.cache_id] = buffer

    @torch.jit.unused
    def _fast_same_check(self, q,k,v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()
        return qkv_same, kv_same
