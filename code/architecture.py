'''
The transformer language model architecture.
'''

import einops
import torch


class Linear(torch.nn.Module):
    """
    A linear layer: y = x.W^T
    dim: ... d_in -> ... d_out
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(self._initialize_weight(device, dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features
        return einops.einsum(
            x, self.weight, 
            "... d_in, d_out d_in -> ... d_out"
        )


    def _initialize_weight(self, device, dtype):
        d_in, d_out = self.in_features, self.out_features
        weight = torch.empty((d_out, d_in), dtype=dtype, device=device)

        mean = 0.
        std = (2 / (d_in + d_out)) ** 0.5
        torch.nn.init.trunc_normal_(weight, mean, std, -3*std, 3*std)
        return weight


class Embedding(torch.nn.Module):
    """
    Embedding layer: token_id(s) -> embedding vector(s)
    dim: vocab_size -> d_model
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.vocab_size = num_embeddings
        self.d_model = embedding_dim

        self.weight = torch.nn.Parameter(self._initialize_weight(device, dtype))


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


    def _initialize_weight(self, device, dtype):
        vocab_size, d_model = self.vocab_size, self.d_model
        weight = torch.empty((vocab_size, d_model), dtype=dtype, device=device)

        mean, std = 0., 1.
        torch.nn.init.trunc_normal_(weight, mean, std, -3., 3.)
        return weight


class RMSNorm(torch.nn.Module):
    """
    RMSNorm for layer normalization: a_i * g_i / sqrt(sum(a_i**2)/d_model + eps)
    dim: ... d_model -> ... d_model
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = torch.nn.Parameter(self._initialize_weight(device, dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model

        in_type = x.dtype
        x = x.to(torch.float32)
        inv_rms = 1./torch.sqrt(torch.sum(x**2, dim=-1)/self.d_model + self.eps)
        
        weight = self.weight.to(torch.float32)
        out = einops.einsum(
            x, weight, inv_rms,
            "... d_model, d_model, ... -> ... d_model"
        )
        return out.to(in_type)


    def _initialize_weight(self, device, dtype):
        return torch.ones((self.d_model,), dtype=dtype, device=device)


class SwiGLU(torch.nn.Module):
    """
    SwiGLU layer: x -> W2 (SiLU(W1 x) ⊙ W3 x)
    dim: ... d_model -> ... d_model
    """

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        assert d_ff is None or d_ff > 0
        if d_ff:
            self.d_ff = d_ff
        else:
            self.d_ff = (d_model * 8 // 3) // 64 * 64 if d_model >= 24 else 64

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model

        w1x = self.w1(x)
        w3x = self.w3(x)

        inner_layer = w1x * torch.sigmoid(w1x) * w3x
        return self.w2(inner_layer)


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Rotary position embeddings.
    dim: ... seq_len d_k -> ... seq_len d_k

    Attributes:
    -----------
        rotations: 
            shape (max_seq_len, d_k/2, 2, 2)
            (max_seq_len, d_k/2) number of 2 by 2 rotaion matrices
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        rotations = self._initialize_rotations(device)
        self.register_buffer("rotations", rotations, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len, d_k = x.shape[-2:]
        assert d_k == self.d_k
        assert seq_len <= self.max_seq_len

        if token_positions is not None:
            assert token_positions.shape[-1] == seq_len
        else:
            token_positions = torch.arange(0, seq_len, device=x.device)

        rotations = self.rotations

        in_type = x.dtype
        x = x.to(torch.float32)
        x = einops.rearrange(
            x, "... (half_d two_in) -> ... half_d two_in", 
            half_d=self.d_k//2, two_in=2
        )

        result = einops.einsum(
            x, rotations[token_positions],
            "... seq_len half_d two_in, ... seq_len half_d two_out two_in -> ... seq_len half_d two_out"
        )
        result = einops.rearrange(
            result, "... half_d two_out -> ... (half_d two_out)",
            half_d=self.d_k//2, two_out=2
        )
        return result.to(in_type)


    def _initialize_rotations(self, device=None):
        positions = torch.arange(0, self.max_seq_len, 1., device=device)
        denominator = torch.pow(
            self.theta, 
            torch.arange(0, self.d_k, 2., device=device) / self.d_k
        )
        angles = einops.einsum(
            positions, 1./denominator, "pos, den -> pos den"
        )

        sin, cos = torch.sin(angles), torch.cos(angles)

        return einops.rearrange(
            torch.stack((cos, -sin, sin, cos)),
            "(two_out two_in) pos half_d -> pos half_d two_out two_in",
            two_in=2, two_out=2
        )


def softmax(x: torch.Tensor, dim: int = -1, temp: float | None = None) -> torch.Tensor:
    """
    Softmax function: v_i -> exp(v_i - max(v)) / sum_j exp(v_j - max(v))
    """
    x_max = torch.amax(x, dim=dim, keepdim=True)

    if temp == 0:
        onehot = (x == x_max).to(x.dtype)
        return onehot / onehot.sum(dim=dim, keepdim=True)

    x_exp = torch.exp(x - x_max if temp is None else (x - x_max) / temp)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None
):
    """
    Attention(Q, K, V) = softmax(Q.K^T / sqrt(d_k)).V
    dim:
        queries: ... n_queries d_k
        keys:    ... n_keys d_k
        values:  ... n_keys d_v
        mask:    n_queries n_keys
        out:     ... n_queries d_v
    """
    assert queries.shape[:-2] == keys.shape[:-2]
    assert queries.shape[-1] == keys.shape[-1]
    n_keys, d_k = keys.shape[-2:]
    n_queries = queries.shape[-2]

    assert keys.shape[:-1] == values.shape[:-1]

    pre_attention = einops.einsum(
        queries, keys,
        "... n_queries d_k, ... n_keys d_k -> ... n_queries n_keys"
    ) / d_k**0.5
    if mask is not None:
        pre_attention = torch.where(mask, pre_attention, other=float("-inf"))

    attention = softmax(pre_attention, dim=-1)
    return einops.einsum(
        values, attention,
        "... n_keys d_v, ... n_queries n_keys -> ... n_queries d_v"
    )


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self attention.
    dim: ... seq_len (num_heads d_k) -> ... seq_len (num_heads d_v)
    """

    def __init__(
        self, d_model: int, num_heads: int, 
        theta: float | None = None, max_seq_len: int = -1, 
        device=None, dtype=None
    ):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        if self.theta:
            assert max_seq_len > 0
            self.rope = RotaryPositionalEmbedding(
                self.theta, self.d_k, self.max_seq_len, device=device
            )

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        assert x.shape[-1] == self.d_model

        seq_len = x.shape[-2]

        num_heads = self.num_heads
        theta = self.theta

        d_k = self.d_k
        d_v = self.d_v

        queries = einops.rearrange(
            self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
            num_heads=num_heads, d_k=d_k
        )
        keys = einops.rearrange(
            self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", 
            num_heads=num_heads, d_k=d_k
        )
        values = einops.rearrange(
            self.v_proj(x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", 
            num_heads=num_heads, d_v=d_v
        )

        if theta:
            if token_positions is not None:
                # token_positions current shape ... seq_len
                # to broadcast, need ... 1 seq_len
                token_positions = token_positions.unsqueeze(dim=-2)

            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        # create mask
        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        mask = torch.tril(mask)

        postattention =  scaled_dot_product_attention(
            queries, keys, values, mask=mask
        )
        postattention = einops.rearrange(
            postattention, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads=num_heads, d_v=d_v
        )
        return self.output_proj(postattention)


class TransformerBlock(torch.nn.Module):
    """
    TransformerBlock: 
    x -> RMSNorm -> Multi-head attention ----> RMSNorm -> SwiGLU FFN -> Output
    └────────────────────────────────────┘  └────────────────────────┘
    """

    def __init__(
            self, d_model: int, num_heads: int, d_ff: int, 
            theta: float | None = None, max_seq_len: int = -1, eps: float = 1e-5, 
            device=None, dtype=None
        ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype
        )

        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y1 = x + self.attn(self.ln1(x), token_positions)
        y2 = y1 + self.ffn(self.ln2(y1))
        return y2


class TransformerLM(torch.nn.Module):
    """
    The transformer language model: (batches of) token_ids -> (batches of) logits
    """

    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int, 
        d_model: int, num_heads: int,
        d_ff: int | None = None, theta: float | None = None, eps: float = 1e-5, 
        device=None, dtype=None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length, eps,
                    device=device, dtype=dtype
                )
            )
        self.ln_final = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        if token_positions is None:
            seq_len = x.shape[-1]
            token_positions = torch.arange(0, seq_len, device=x.device)

        x = self.token_embeddings(x)

        for block in self.layers:
            x = block(x, token_positions)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x


@torch.inference_mode()
def decode(
    model, inputs, max_tokens, 
    ending_token=None, temp=1., p=1.
):

    assert len(inputs) > 0

    context_length = model.context_length
    device = next(model.parameters()).device

    tokens = inputs.copy()

    for _ in range(max_tokens):

        x = torch.tensor(tokens[-context_length:], device=device)

        if temp == 0:
            next_token = torch.argmax(probs, dim=-1).item()
        else:
            probs = softmax(model(x)[-1], temp=temp)
            probs, indices = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(probs, dim=-1)

            if p >= 1:
                idx = len(probs)
            else:
                idx = torch.searchsorted(cumsum, p)

            next_token = torch.multinomial(probs[:idx+1], num_samples=1)
            next_token = indices[next_token].item()

        tokens.append(next_token)

        if next_token == ending_token:
            break

    return tokens