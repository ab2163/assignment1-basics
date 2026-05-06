import torch
import torch.nn as nn
import math
from einops import einsum
from einops import reduce
from einops import rearrange
from jaxtyping import Float, Int, Bool
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        stddev = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=stddev, a=-3*stddev, b=3*stddev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in_f, out_f in_f -> ... out_f")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        stddev = 1
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=stddev, a=-3*stddev, b=3*stddev)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (reduce(x.pow(2), "... d_model -> ... 1", "mean") + self.eps).sqrt()
        x_normed = x / rms
        return (self.gain * x_normed).to(original_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        # d_ff ≈ (8/3) * d_model, rounded up to nearest multiple of 64
        if d_ff is None:
            d_ff = math.ceil((8/3 * d_model) / 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(w1(x)) * w3(x))
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)  # SiLU(w1(x))
        return self.w2(silu * self.w3(x))  # GLU: element-wise gate * branch

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        # compute frequencies for each pair: shape (d_k/2,)
        k = torch.arange(0, d_k, 2, device=device).float()
        freqs = 1.0 / (theta ** (k / d_k))

        # compute angles for all positions: shape (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, device=device).float()
        angles = einsum(positions, freqs, "seq, d_half -> seq d_half")

        # precompute and store cos/sin tables
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)

        # look up cos/sin for the given positions
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)

        # split x into pairs
        x_even, x_odd = rearrange(x, "... seq (d_half two) -> two ... seq d_half", two=2)

        # apply 2D rotation to each pair
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd  = x_even * sin + x_odd * cos

        # interleave back together
        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return rearrange(x_out, "... seq d_half two -> ... seq (d_half two)")

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # subtract max for numerical stability
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "... seq_len d_v"]:

    d_k = Q.shape[-1]

    # compute attention scores: (..., seq_len, seq_len)
    scores = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / (d_k ** 0.5)

    # apply mask: set False positions to -inf
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # softmax over key dimension
    attn_weights = softmax(scores, dim=-1)

    # weighted sum of values: (..., seq_len, d_v)
    return einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, use_rope: bool = True, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"], 
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        seq_len = x.shape[-2]

        # project to Q, K, V: (..., seq_len, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # split into heads: (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        K = rearrange(K, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)
        V = rearrange(V, "... seq (h d_k) -> ... h seq d_k", h=self.num_heads)

        # apply RoPE to Q and K (not V), heads are batch dims
        if self.use_rope:
            token_positions_expanded = rearrange(token_positions, "... seq -> ... 1 seq")
            Q = self.rope(Q, token_positions_expanded)
            K = self.rope(K, token_positions_expanded)

        # causal mask: (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # attention: (..., num_heads, seq_len, d_k)
        # h treated like a batch dimension so you apply attention individually to each head
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)

        # concatenate heads: (..., seq_len, d_model)
        attn_out = rearrange(attn_out, "... h seq d_k -> ... seq (h d_k)")

        # final projection
        return self.w_o(attn_out)

class FFNSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        # d_ff = 4 * d_model to match SwiGLU parameter count
        if d_ff is None:
            d_ff = 4 * d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN_SiLU(x) = W2 * SiLU(W1 * x)
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)
        return self.w2(silu)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_norm: bool = True,
        pre_norm: bool = True,  # True = pre-norm, False = post-norm
        device=None,
        dtype=None,
        use_rope = True,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.pre_norm = pre_norm

        if use_norm:
            self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
            self.ff_norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope, device=device, dtype=dtype)

        # choose FFN type
        if use_swiglu:
            self.ff = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ff = FFNSiLU(d_model, device=device, dtype=dtype)  # uses 4*d_model internally

    def forward(self, x, token_positions=None):
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)

        # first sub-layer: attention
        if self.use_norm and self.pre_norm:
            # pre-norm: norm -> attn -> residual
            x = x + self.attn(self.attn_norm(x), token_positions)
        elif self.use_norm and not self.pre_norm:
            # post-norm: attn -> residual -> norm
            x = self.attn_norm(x + self.attn(x, token_positions))
        else:
            # no norm
            x = x + self.attn(x, token_positions)

        # second sub-layer: feedforward
        if self.use_norm and self.pre_norm:
            x = x + self.ff(self.ff_norm(x))
        elif self.use_norm and not self.pre_norm:
            x = self.ff_norm(x + self.ff(x))
        else:
            x = x + self.ff(x)

        return x
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
        use_norm: bool = True, 
        pre_norm: bool = True,
        use_rope = True,
        use_swiglu = True,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, theta, use_norm=use_norm,
                pre_norm=pre_norm, device=device, dtype=dtype, use_rope=use_rope, use_swiglu=use_swiglu)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length

    def forward(
        self,
        token_ids: Int[Tensor, "batch seq_len"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len vocab_size"]:

        if token_positions is None:
            seq_len = token_ids.shape[-1]
            token_positions = torch.arange(seq_len, device=token_ids.device)

        # embed tokens
        x = self.embedding(token_ids)

        # pass through transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # final norm and LM head
        x = self.final_norm(x)
        return self.lm_head(x)
    
def cross_entropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    # subtract max for numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values

    # target logits: index into logits at the target class for each position
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # log(sum(exp(logits))) - computed stably since we already subtracted max
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))

    # cross entropy: -target_logit + log_sum_exp, averaged over all batch dims
    loss = -target_logits + log_sum_exp
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # initialise state for this parameter if first step
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)  # first moment
                    state["v"] = torch.zeros_like(p.data)  # second moment

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # update biased moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

                # update parameters
                p.data.mul_(1 - lr * weight_decay)                      # weight decay
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)     # gradient step

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    
    # warmup phase
    if t < T_w:
        return alpha_max * (t / T_w)
    
    # cosine decay phase
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * progress))
    
    # after schedule ends, hold at minimum
    else:
        return alpha_min

def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6) -> None:
    # collect all gradients that exist
    grads = [p.grad for p in parameters if p.grad is not None]

    # compute global l2 norm across all parameters
    global_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))

    # scale down if norm exceeds max_norm
    if global_norm > max_norm:
        scale = max_norm / (global_norm + eps)
        for g in grads:
            g.mul_(scale)