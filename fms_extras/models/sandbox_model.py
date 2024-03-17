import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.models.llama import LLaMAConfig
from fms.modules.embedding import WordEmbedding
from fms.utils.activation import str_to_activation


def pscan(X, A):
    # Courtesy of https://github.com/pytorch/pytorch/issues/95408#issuecomment-1871722410
    Xa = F.pad(torch.transpose(X, 1, 2), (1,0))
    
    X_ = Xa.add(1e-12).log()
    A_ = A.log()
#     X_real = torch.abs(Xa).log()
#     X_complex = (Xa < 0).to(A.dtype)
#     A_real = torch.abs(A).log()
#     X_ = torch.complex(X_real, X_complex * torch.pi)
#     A_complex = (A < 0).to(A.dtype)
#     A_ = torch.complex(A_real, A_complex * torch.pi)

    a_star =  F.pad(torch.cumsum(A_, dim=1).transpose(1,2), (1,0))
    log_x0_plus_b_star = torch.logcumsumexp(X_ - a_star, dim=-1)
    log_x =  a_star + log_x0_plus_b_star
    return torch.transpose(torch.exp(log_x)[:,:,1:], 1, 2)


def scan(state, g):
    # b n d
    state = torch.stack([state, g], dim=1) # b 2 n d
    logl = state.size(2).bit_length() - 1
    s = state.size()
    # Up sweep: create ruler ticks
    for i in range(logl):
        span = 2**(i+1)
        s_adjust = [s[0], 2, -1, span, s[3]]
        state = state.view(*s_adjust)
        state[:,0,:,-1] = torch.addcmul(state[:,0,:,-1], state[:,0,:,span//2-1], state[:,1,:,-1])
        state[:,1,:,-1] *= state[:,1,:,span//2-1]
        
    # Down sweep: fill in blanks
    state = state.view(*s)
    state = nn.functional.pad(state, (0,0,1,0))
    remainder = state[:,:,-1:]
    state = state[:,:,:-1]
    for i in range(logl-1):
        span = 2**(logl-i-1)
        s_adjust = [s[0], 2, -1, span, s[3]]
        state = state.view(*s_adjust)
        state[:,0,:,span//2] = torch.addcmul(state[:,0,:,span//2], state[:,0,:,0], state[:,1,:,span//2])
        state[:,1,:,span//2] *= state[:,1,:,0]
    state = torch.cat([state.view(*s)[:,:,1:], remainder], dim=-2)
    return state[:,0]


class GatedScan(torch.autograd.Function):
    @staticmethod
    def forward(state, gate):
        return pscan(state.mul(1 - gate), gate)

    @staticmethod
    def setup_context(ctx, inputs, output):
        state, gate = inputs
        ctx.save_for_backward(state, gate, output)

    @staticmethod
    def backward(ctx, grad):
        state, gate, output = ctx.saved_tensors

        # Gate-accumulate grads
        gflip = gate.flip([1])
        gatesum = pscan(grad.flip([1]), gflip.roll(1, dims=1)).flip([1])

        # State grad
        state_grad = gatesum.mul(1 - gate)

        # Gate grad
        outshift = output.roll(1, dims=1)
        outshift[:, :1] = 0
        gate_grad = gatesum.mul(outshift - state)

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


# class SandboxUnit(nn.Module):
#     def __init__(
#         self,
#         emb_dim,
#         hidden_grow_factor=1.5,
#         multiple_of=None,
#         activation_fn=nn.ReLU(),
#         p_dropout=0.1,
#         use_bias=False,
#         ln_eps=1e-6,
#     ):
#         super(SandboxUnit, self).__init__()
#         self.multiple_of = multiple_of
#         hidden_dim = int(hidden_grow_factor * emb_dim)
#         if multiple_of:
#             hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
#             hidden_grow_factor = hidden_dim / emb_dim
#         self.hidden_dim = hidden_dim
#         self.w_in = nn.Linear(emb_dim, hidden_dim * 3, bias=use_bias)
#         self.bias = nn.Parameter(torch.rand(hidden_dim))
#         # self.mulbias = nn.Parameter(torch.zeros(hidden_dim))
#         # self.conv = nn.Parameter(torch.zeros(emb_dim))
#         self.a = activation_fn
#         self.p_dropout = p_dropout
#         if p_dropout:
#             self.d = nn.Dropout(p_dropout)
#         self.w_out = nn.Linear(hidden_dim, emb_dim, bias=use_bias)
#         self.use_bias = use_bias
#         self.width = emb_dim
#         self.hidden_grow_factor = hidden_grow_factor
#         self.scan = GatedScan.apply
#         self.ln = CenteredLayerNormParameterized(emb_dim, eps=ln_eps, use_high_precision_pow=True)
#         # self.layer_bias = 0

#     def reset_parameters(self, gain=1.0):
#         # Gain for init scale factor x is given by:
#         # (q / sqrt2) * v * wout
#         # Plugging in:
#         # x sqrtd / sqrt2 * x sqrtd * x sqrtd sqrtg
#         # Set to gain, solve for x
#         # x**3 (sqrt.5 * sqrtg * d**1.5) = target gain
#         # x = (gain * sqrt2 / d**1.5 / sqrtg)**(1/3)
#         for layer in [self.w_in.weight, self.w_out.weight]:  # , self.conv, self.mulbias]:
#             nn.init.normal_(
#                 layer,
#                 mean=0.0,
#                 std = (gain * 2**.5 / self.hidden_grow_factor**.5 / self.width**1.5)**(1/3)
#             )
#         self.bias.data.random_()
#         if self.use_bias:
#             self.w_in.bias.data.zero_()
#             self.w_out.bias.data.zero_()
#         self.ln.reset_parameters()

#     def forward(self, x):
#         # x: b n d
#         # TODO: add dropout somewhere
#         residual = x

#         # Conv, RWKV-style
#         # c = self.conv
#         # c = c.add(self.layer_bias).sigmoid()
#         # x = x * c + (1 - c) * F.pad(x, (0,0,1,0))[:,:-1]

#         # Layernorm
#         x = self.ln(x)

#         # Project
#         q, g, v = self.w_in(x).split(self.hidden_dim, dim=-1)
#         q = self.a(q)
#         # v = self.a(v)

#         # Gate handling
#         b = self.bias
#         b = b.sub(b.min())
#         b = b.mul(6 / b.max())
#         g = g.add(b).sigmoid()
        
#         # Scan
#         z = self.scan(v, g)#.add(self.mulbias)

#         # Out project / add
#         return residual + self.w_out(z * q)
    

class SandboxUnit(nn.Module):
    def __init__(
        self,
        emb_dim,
        nheads=8,
        head_dim=8,
        hidden_grow_factor=1.5,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=False,
        ln_eps=1e-6,
    ):
        super(SandboxUnit, self).__init__()

        # v: d/h
        # k: e
        # q: h*e
        # g: d/h
        # z: d

        self.multiple_of = multiple_of
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            hidden_grow_factor = hidden_dim / emb_dim
        assert hidden_dim % nheads == 0 and hidden_dim % head_dim == 0
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.headdim = head_dim
        self.inner_dims = [hidden_dim//nheads, head_dim, nheads*head_dim, hidden_dim//nheads, hidden_dim]
        self.w_in = nn.Linear(emb_dim, sum(self.inner_dims), bias=use_bias)
        self.bias = nn.Parameter(torch.rand(hidden_dim//nheads, head_dim))
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w_out = nn.Linear(hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.hidden_grow_factor = hidden_grow_factor
        self.scan = GatedScan.apply
        self.ln = CenteredLayerNormParameterized(emb_dim, eps=ln_eps, use_high_precision_pow=True)
        # self.layer_bias = 0

    def reset_parameters(self, gain=1.0):
        # Gain for init scale factor x is given by:
        # (z / sqrt2) * v * k / sqrt2 * q * wout
        # Plugging in:
        # x sqrtd / sqrt2 * x sqrtd * x sqrtd / sqrt2 * x sqrtd sqrte * x sqrtd sqrtg
        # Set to gain, solve for x
        # x**5 (.5 * sqrtg * sqrte * d**2.5) = target gain
        # x = (gain * 2 / d**2.5 / sqrtge)**(1/5)
        for layer in [self.w_in.weight, self.w_out.weight]:
            nn.init.normal_(
                layer,
                mean=0.0,
                std = (gain * 2 / (self.hidden_grow_factor*self.headdim)**.5 / self.width**2.5)**(1/5)
            )
        self.bias.data.random_()
        if self.use_bias:
            self.w_in.bias.data.zero_()
            self.w_out.bias.data.zero_()
        self.ln.reset_parameters()

    def forward(self, x):
        # x: b n d
        # TODO: add dropout somewhere
        residual = x

        # Layernorm
        x = self.ln(x)

        # Project
        v, k, q, g, z = self.w_in(x).split(self.inner_dims, dim=-1)
        z = self.a(z)

        # Gate handling
        b = self.bias
        b = b.sub(b.min())
        b = b.mul(6 / b.max())
        g = g.unsqueeze(-1).add(b).sigmoid() # b n d/h e
        s = g.size()
        
        # Expand state
        kv = v.unsqueeze(-1) * k.unsqueeze(-2) # b n d/h e
        
        # Scan
        kv = self.scan(kv.view(*s[:2],-1).relu(), g.view(*s[:2],-1)) # b n d/h*e
        qkv = torch.einsum("bnde,bnhe->bnhd", 
                           kv.view(*s), 
                           q.view(*s[:2], self.nheads, self.headdim)
                           ).reshape(*s[:2], self.hidden_dim)

        # Out project / add
        return residual + self.w_out(z * qkv)


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
                self.config.emb_dim,
                hidden_grow_factor=self.config.hidden_grow_factor,
                multiple_of=self.config.multiple_of,
                activation_fn=str_to_activation(self.config.activation_fn),
                p_dropout=self.config.p_dropout,
                use_bias=False,
                ln_eps=self.config.norm_eps,
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
            layer.reset_parameters(gain=1/len(self.layers)**.5)
            # layer.layer_bias = layer_ind * 3 / len(self.layers)

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
