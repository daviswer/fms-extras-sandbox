import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.models.llama import LLaMAConfig
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.utils.activation import str_to_activation


def scan(state, g):
    # state: b n d h
    # g: b n h h
    print(state.size(), g.size())
    state = state.clone()
    g = g.clone()
    s = state.size()
    logl = s[1].bit_length() - 1
    # Up sweep: create ruler ticks
    for i in range(logl):
        span = 2**(i+1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(s[0], -1, span, s[3], s[3])  # b -1 span h h
        newstate = state[:,:,span//2-1].matmul(g[:,:,-1])
        newgate = g[:,:,span//2-1].matmul(g[:,:,-1])
        print("   ", newstate.shape, newgate.shape)
        state[:,:,-1] += newstate
        g[:,:,-1] = newgate
        
    # Down sweep: fill in blanks
    state = state.view(*s)
    g = g.view(s[0], s[1], s[3], s[3])
    state = nn.functional.pad(state, (0,0,0,0,1,0))
    g = nn.functional.pad(g, (0,0,0,0,1,0))[:,:-1]
    remainder = state[:,-1:]
    state = state[:,:-1]
    for i in range(logl-1):
        span = 2**(logl-i-1)
        state = state.view(s[0], -1, span, *s[2:])  # b -1 span d h
        g = g.view(s[0], -1, span, s[3], s[3])  # b -1 span h h
        state[:,:,span//2] = state[:,:,span//2] + state[:,:,0].matmul(g[:,:,span//2])
        g[:,:,span//2] = g[:,:,0].matmul(g[:,:,span//2])
    state = torch.cat([state.view(*s)[:,1:], remainder], dim=1)
    return state


class MatScan(torch.autograd.Function):
    @staticmethod
    def forward(state, gate):
        return scan(state, gate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        state, gate = inputs
        ctx.save_for_backward(state, gate, output)

    @staticmethod
    def backward(ctx, grad):
        state, gate, output = ctx.saved_tensors

        # Gate-accumulate grads
        gflip = gate.flip([1]).transpose(2,3)
        gatesum = scan(grad.flip([1]), gflip.roll(1, dims=1)).flip([1])  # b n d h

        # State grad
        state_grad = gatesum #.mul(1 - gate)

        # Gate grad
        outshift = output.roll(1, dims=1)
        outshift[:, :1] = 0  # b n d h
        gate_grad = outshift.transpose(2,3).matmul(gatesum)  # b n h h

        return state_grad, gate_grad


class CenteredLayerNormParameterized(nn.Module):
    """
    As LayerNormParameterized from Foundation-Model-Stack, but adds affine weight values to a static offset of 1.
    This allows us to use L2-normalization without issue.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(CenteredLayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        
    def reset_parameters(self):
        if self.elementwise_scale:
            self.weight.data.zero_()
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x):
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight.add(1) * x
        if self.elementwise_shift:
            x = x + self.bias
        return x
    

class ScanHeadAttention(nn.Module):
    """
    Performs scan-cache self-attention.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
    ):
        super(ScanHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        g = torch.tensor([1 - 2**(-i/32*5-1) for i in range(32)])
        gates = torch.diag(g)
        for i in range(1, g.size(0)):
            gates[i,i-1] = (1-g[i])
        self.register_buffer("gates", gates)
        self.scan = MatScan.apply

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self,
        q,
        k,
        v,
    ):
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )
        keys = self.key(k)
        values = self.value(v)
        queries = queries / (self.emb_kq_per_head**(1/4))  # b l d h
        keys = keys / (self.emb_kq_per_head**(1/4))
        
        # Build scan cache
        # k/v: b l d
        gate = self.gates[None,None]  # 1 1 32 32
        gate = gate.expand(batch_size, kv_len, -1, -1)  # b l 32 32
        keys = keys.unsqueeze(3)  # b l d 1
        keys = F.pad(keys, (0, 32-1))  # b l d 32
        keys = self.scan(keys, gate).view(batch_size, kv_len, self.kvheads, self.emb_kq_per_head, -1)  # b l h d 32
        values = values.unsqueeze(3)  # b l d 1
        values = F.pad(values, (0, self.kvheads-1))  # b l d 32
        values = self.scan(values, gate).view(batch_size, kv_len, self.kvheads, self.emb_v_per_head, -1)  # b l h d 32

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b l h d 32
        if expansion != 1:
            keys = keys.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
            values = (
                values.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
            )
        
        # q/k/v: b h l d
        qk = torch.einsum("blhd,blhde->blhe", queries, keys)  # b l h 32
        qk = qk.softmax(3)
        qkv = torch.einsum("blhe,blhde->blhd", qk, values)  # b l h d

        z = qkv.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        return self.dense(z)
    

class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self,
        q,
        k,
        v,
    ):
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )
        keys = self.key(k).view(
            batch_size, kv_len, self.kvheads, self.emb_kq_per_head
        )
        values = self.value(v).view(
            batch_size, kv_len, self.kvheads, self.emb_v_per_head
        )
        queries = queries.transpose(2, 1) / (self.emb_kq_per_head**(1/4))
        keys = keys.transpose(2, 1) / (self.emb_kq_per_head**(1/4))
        values = values.transpose(2, 1)  # compatible with QK.T
        
        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b h l d
        if expansion != 1:
            keys = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values = (
                values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            )
        
        # q/k/v: b h l d
        qk = queries.matmul(keys.transpose(2,3))  # b h l l
        m = torch.ones(qk.size(2), qk.size(3), device=qk.device, dtype=qk.dtype).tril().log()
        qk = qk.add(m).softmax(3)
        qkv = qk.matmul(values)  # b h l d

        z = qkv.transpose(1,2).reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        return self.dense(z)



class SandboxUnit(nn.Module):
    def __init__(
        self,
        config
    ):
        super(SandboxUnit, self).__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        self.ln = CenteredLayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = CenteredLayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = ScanHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,
        )


    def forward(self, x):
        # x: b n d
        residual = x
        x = self.ln(x)
        x = self.attn(x,x,x)
        x = x + residual

        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        x = x + residual

        return x


class SandboxModel(nn.Module):
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(SandboxModel, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        shared = WordEmbedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
            abs_pos=False,
            reversible=True,
            tie_weights=False,
            bias=False,
        )
        self.shared = self.distributed_strategy.distribute_module(shared)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = SandboxUnit(
                self.config
            )
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = CenteredLayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def get_config(self) -> LLaMAConfig:
        return self.config

    @classmethod
    def from_config(cls, config: LLaMAConfig) -> "SandboxModel":
        return cls(config)

    def reset_parameters(self):
        nn.init.normal_(self.shared.emb.weight, std=1/self.width**.5)
        # self.shared.head.weight.data.zero_()
        nn.init.normal_(self.shared.head.weight, std=1/self.width**.5)
        self.dec_norm.reset_parameters()
        for layer_ind, layer in enumerate(self.layers):
            layer.ln.reset_parameters()
            layer.ff_ln.reset_parameters()
            layer.attn.reset_parameters()
            layer.ff_sub_layer.reset_parameters()

    def _helper(
        self,
        x_in,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm=None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        qlen = x_in.size(1)
        klen = x_in.size(1)

        # TODO: CACHING NOT YET SUPPORTED
        # # if we are using the cache, the key length needs to be extended with the past keys length
        # if use_cache and past_key_value_states[0] is not None:
        #     klen += past_key_value_states[0][0].size(-2)

        # # if mask is none, we need to specify causal mask
        # if mask is None:
        #     # we are caching and can assume all 1s in the mask
        #     if use_cache and klen != 1 and qlen == 1:
        #         # b x h x qlen x kvlen
        #         is_causal_mask = False
        #     else:
        #         is_causal_mask = True
        # else:
        #     is_causal_mask = False

        x_in = self.shared(x_in).mul(self.width**.5)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                # mask=mask,
                # position_ids=position_ids,
                # past_key_value_state=past_key_value_states[i],
                # use_cache=use_cache,
                # is_causal_mask=is_causal_mask,
                # attn_algorithm=attn_algorithm,
            )

            # if use_cache:
            #     x_in, present_key_value_state = output
            #     present_key_value_states.append(present_key_value_state)

            # else:
            #     x_in = output
            x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states

    def forward(
        self,
        x,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        only_last_token=False,
        attn_algorithm=None,
    ):
        output, cache = self._helper(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.shared(output, reverse=True)

        if use_cache:
            return preds, cache
        else:
            return preds