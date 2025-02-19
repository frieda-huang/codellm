from dataclasses import dataclass
from typing import List, Optional, Self, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

MaskCache = torch.Tensor
RoPECache = torch.Tensor
KVCache = Tuple[torch.Tensor, torch.Tensor]


def find_multiple(n: int, k: int) -> int:
    """
    n % k gives us how far we are past the last multiple
    k - (n % k) tells us how much we need to add to get to the next multiple
    adding k - (n % k) to n gets us to the next multiple of k
    """
    if n % k == 0:  # if n is already a multiple of k
        return n
    return n + k - (n % k)  # round up to next multiple of k


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
    # Small config for M2 MacBook
    "tiny": dict(n_layer=8, n_head=8, n_embd=512),
}


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        # project hidden states to vocabulary probabilities
        # (n_embd) -> (vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        self.transformer = nn.ModuleDict(
            dict(
                # token embeddings -> convert token ids to vectors
                # (vocab_size) -> (n_embd)
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                # stack of transformer blocks
                # create n_layer blocks of transformers
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                # final layer normalization using RMS normalization
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None  # for position embeddings
        self.mask_cache: Optional[MaskCache] = None  # for attention masks
        self.kv_cache: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        # this method is passed to the model's .apply() function, which recursively applies it to each submodule, ensuring consistant initialization
        if isinstance(module, nn.Linear):
            # 0.02 and `2 * self.config.n_layer` are heuristics to control the variance as layers are stacked in deep transformer models
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
                * torch.rsqrt(torch.tensor(2 * self.config.n_layer, dtype=torch.float)),
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
                * torch.rsqrt(torch.tensor(2 * self.config.n_layer, dtype=torch.float)),
            )

    def forward(
        self,
        idx: torch.Tensor,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert (
            T <= max_seq_length
        ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            T <= block_size
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        # handle position embeddings and attention masking differently
        # based on whether we are doing token-by-token generation or
        # processing a full sequence
        if input_pos is not None:
            # we're selecting from the sequence positions in the first dimension
            rope = self.rope_cache.index_select(0, input_pos)
            # we're selecting from the query sequence positions in the third dimension
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            # used during training
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        x = self.transformer.wte(idx)  # (B, T, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            # used during training
            # each transformer block processes the input without caching anything
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_cache:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                # create kv caches for each layer
                self.kv_cache = [
                    (
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                    )
                    for _ in range(self.config.n_layer)
                ]
            # process each token while maintaining the kv caches
            # the block returns both the processed tokens and updated caches
            # caches store previously computed k,v values to avoid recomputing them
            for i, block in enumerate(self.transformer.h):
                x, self.kv_cache[i] = block(
                    x, rope, mask, max_seq_length, input_pos, self.kv_cache[i]
                )

        x = self.transformer.ln_f(x)

        # convert to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)


# standard transformer block
class Block(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        # self-attention mechanism with causal masking
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,  # input tensor containing token embeddings (batch_size, seq_len, dim)
        rope: RoPECache,  # (batch_size, dim/2, 2)
        mask: MaskCache,  # (batch_size, seq_len, seq_len)
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # h: (batch_size, seq_len, dim)
        # new_kv_cache: (batch_size, seq_len, n_heads, dim/n_heads)
        h, new_kv_cache = self.attn(
            self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache
        )
        # first residual connections after attention
        x = x + h
        # second residual connection after mlp
        x = x + self.mlp(self.rms_2(x))  # (batch_size, seq_len, dim)
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[
            torch.Tensor
        ] = None,  # contain the positions where new tokens should be inserted to kv_cache
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # c_attn projects input to concatenated Q,K,V
        # c_attn weight matrix shape: (C, 3*C)
        # output shape: (B, T, 3*C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # after split, each of q,k,v has shape: (B, T, C)

        head_size = C // self.n_head

        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        # shape remains same (B, T, nh, hs)
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        # this only happens during inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] > max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift one position to the left along the seq_len dimension
                cache_k = torch.roll(cache_k, shifts=-1, dims=2)
                cache_v = torch.roll(cache_v, shifts=-1, dims=2)

            k = torch.index_copy(cache_k, index=input_pos, source=k)
            v = torch.index_copy(cache_v, index=input_pos, source=v)

            kv_cache = k, v

        # no dropout is being applied to the attention weights, which is common during inference
        # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # reshape back to original dimensions
        y = y.transpose(1, 2)  # (B, T, nh, hs)

        # `.contiguous()` forces the tensor to have a contiguous memory layout
        # `.view()` reshapes the tensor to dimensions (B, T, C) and combines the n_head and head_size back to a single dimension C
        y = y.contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y, kv_cache


class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    # Based on the paper *https://arxiv.org/pdf/1910.07467v1

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        # dim: embedding size, eps: small value to avoid division by zero
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # .mean(-1, keepdim=True) computes the mean across the last dimension, i.e. feature space
        norm_x = x.pow(2).mean(-1, keepdim=True)

        # NOTE: torch.rsqrt(x) is faster and more numerically stable than 1 / torch.sqrt(x)
        # (batch_size, seq_len, dim) x (batch_size, seq_len, 1) = (batch_size, seq_len, dim)
        return x * torch.rsqrt(norm_x + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x.float()` converts x to `torch.float32` to prevent issues if x was originally an integer (torch.int64)
        # RMSNorm must use floating-point numbers since it involves division and square root operations
        return self.gamma * self._norm(x.float()).type_as(x)


# RoPE implementation
# key ideas are
# 1. each embedding dimension is treated as pairs of numbers (real/imaginary)
# 2. each position has a unique rotation angle that scales with the position index
# 3. different dimension pairs rotate at different frequencies (controlled by θₖ)
# 4. the rotation is implemented as complex multiplication using cached trigonometric values
# for in-depth explaination, go to https://www.youtube.com/watch?v=oM4VmoabDAI&t=80s
def build_rope_cache(
    seq_len: int,  # Maximum sequence length
    n_elem: int,  # Hidden dimension size (must be even)
    dtype: torch.dtype,
    device: torch.device,
    base: int = 1000,  # Base for frequency computation
) -> RoPECache:
    """build a cache of RoPE angles and trigonometric functions.

    this implements the RoPE (Rotary Position Embedding) frequency computation and caching.
    for each dimension pair i, we compute a frequency θᵢ that decreases geometrically,
    allowing the model to capture both fine and coarse position dependencies.
    """
    # 1. compute frequencies θᵢ for each dimension pair
    # θᵢ = 1/base^(2i/d) where i is the dimension index
    # shape: (n_elem/2,) - one frequency per dimension pair
    theta = 1.0 / (
        base ** (torch.arange(1, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # 2. generate position indices [0, 1, ..., seq_len-1]
    # shape: (seq_len,)
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # 3. compute mθ for all combinations of positions and frequencies
    # shape: (seq_len, n_elem/2)
    # each element [m,k] = m·θₖ, representing the rotation angle
    # for position m and dimension pair k
    idx_theta = torch.outer(seq_idx, theta).float()

    # 4. compute and cache cos(mθ) and sin(mθ) for each position and frequency
    # shape: (seq_len, n_elem/2, 2)
    # last dimension stores [cos(mθ), sin(mθ)] for each position m and frequency θ
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # 5. handle reduced precision datatypes
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """apply rotary position embeddings to input tensor.

    args:
        x: input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        rope_cache: Cached rotary embeddings of shape (seq_len, head_dim/2, 2)

    returns:
        tensor with rotary embeddings applied, same shape as input
    """
    # 1. handle variable sequence lengths by truncating cache
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # 2. reshape input to separate each dimension into real/imaginary pairs
    # input shape: (batch_size, seq_len, n_heads, head_dim)
    # reshaped: (batch_size, seq_len, n_heads, head_dim/2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # 3. reshape cache for broadcasting
    # from: (seq_len, head_dim/2, 2)
    # to: (1, seq_len, 1, head_dim/2, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

    # 4. apply rotary embeddings using complex multiplication:
    # for each position m and dimension k:
    # (x + yi) * e^(imθ) = (x + yi)(cos(mθ) + isin(mθ))
    # = (x·cos(mθ) - y·sin(mθ)) + i(x·sin(mθ) + y·cos(mθ))
    x_out2 = torch.stack(
        [
            # real part: x·cos(mθ) - y·sin(mθ)
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            # imaginary part: x·sin(mθ) + y·cos(mθ)
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    # 5. restore original shape by flattening the last two dimensions
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
