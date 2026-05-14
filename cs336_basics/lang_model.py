import torch
import torch.nn as nn
import math
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl
from einops import einsum
from einops import reduce
from einops import rearrange
from jaxtyping import Float, Int, Bool
from torch import Tensor
from torch.utils.checkpoint import checkpoint

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

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    nvtx.range_push("attn_scores_matmul")
    scores = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / (d_k ** 0.5)
    nvtx.range_pop()

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    nvtx.range_push("attn_softmax")
    attn_weights = softmax(scores, dim=-1)
    nvtx.range_pop()

    nvtx.range_push("attn_values_matmul")
    result = einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    nvtx.range_pop()

    return result

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

        with nvtx.range("transformer_block"):
            # first sub-layer: attention
            with nvtx.range("attn_sublayer"):
                if self.use_norm and self.pre_norm:
                    # pre-norm: norm -> attn -> residual
                    with nvtx.range("attn_norm"):
                        normed = self.attn_norm(x)
                    with nvtx.range("attn"):
                        attn_out = self.attn(normed, token_positions)
                    x = x + attn_out
                elif self.use_norm and not self.pre_norm:
                    # post-norm: attn -> residual -> norm
                    with nvtx.range("attn"):
                        attn_out = self.attn(x, token_positions)
                    with nvtx.range("attn_residual"):
                        residual = x + attn_out
                    with nvtx.range("attn_norm"):
                        x = self.attn_norm(residual)
                else:
                    # no norm
                    with nvtx.range("attn"):
                        x = x + self.attn(x, token_positions)

            # second sub-layer: feedforward
            with nvtx.range("ffn_sublayer"):
                if self.use_norm and self.pre_norm:
                    with nvtx.range("ffn_norm"):
                        normed = self.ff_norm(x)
                    with nvtx.range("ffn"):
                        ffn_out = self.ff(normed)
                    x = x + ffn_out
                elif self.use_norm and not self.pre_norm:
                    # post-norm: ffn -> residual -> norm
                    with nvtx.range("ffn"):
                        ffn_out = self.ff(x)
                    with nvtx.range("ffn_residual"):
                        residual = x + ffn_out
                    with nvtx.range("ffn_norm"):
                        x = self.ff_norm(residual)
                else:
                    with nvtx.range("ffn"):
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
        use_checkpoint: bool = False,
        checkpoint_blocks: int = 1,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
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
        if self.use_checkpoint and self.checkpoint_blocks > 1:
            # checkpoint every k blocks
            def run_k_blocks(x, start_idx):
                for i in range(start_idx, min(start_idx + self.checkpoint_blocks, len(self.layers))):
                    x = self.layers[i](x, token_positions)
                return x

            for start in range(0, len(self.layers), self.checkpoint_blocks):
                x = checkpoint(
                    run_k_blocks, x, start,
                    use_reentrant=False
                )
        elif self.use_checkpoint:
            # checkpoint every block
            for layer in self.layers:
                x = checkpoint(layer, x, token_positions, use_reentrant=False)
        else:
            # no checkpointing
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

import torch
import torch.nn.functional as F
from einops import rearrange
import math

class FlashAttentionPyTorch(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass in pure PyTorch.
        
        Args:
            Q: (batch, seq_len, d_k)
            K: (batch, seq_len, d_k)
            V: (batch, seq_len, d_v)
            is_causal: bool (ignored for now)
        
        Returns:
            O: (batch, seq_len, d_v)
        """
        batch, seq_len, d_k = Q.shape
        d_v = V.shape[-1]
        scale = 1.0 / math.sqrt(d_k)

        # tile sizes — at least 16x16
        BLOCK_SIZE = max(16, min(64, seq_len))

        # output and logsumexp
        O = torch.zeros(batch, seq_len, d_v, device=Q.device, dtype=Q.dtype)
        L = torch.full((batch, seq_len), float('-inf'), device=Q.device, dtype=torch.float32)

        # number of tiles
        num_tiles = seq_len // BLOCK_SIZE

        for i in range(num_tiles):
            # query tile indices
            q_start = i * BLOCK_SIZE
            q_end   = q_start + BLOCK_SIZE

            # load query tile
            Q_i = Q[:, q_start:q_end, :]  # (batch, BLOCK_SIZE, d_k)

            # running statistics for this query tile
            # running maximums (for each BLOCK_SIZE the inner loop goes over)
            m_i = torch.full((batch, BLOCK_SIZE), float('-inf'), 
                           device=Q.device, dtype=torch.float32)
            # running sums (for each BLOCK_SIZE the inner loop goes over)
            d_i = torch.zeros(batch, BLOCK_SIZE, 
                            device=Q.device, dtype=torch.float32)
            # "accumulator" (i.e. the slice of the output matrix you are computing)
            O_i = torch.zeros(batch, BLOCK_SIZE, d_v, 
                            device=Q.device, dtype=Q.dtype)

            for j in range(num_tiles):
                # key/value tile indices
                kv_start = j * BLOCK_SIZE
                kv_end   = kv_start + BLOCK_SIZE

                # load key/value tiles
                K_j = K[:, kv_start:kv_end, :]  # (batch, BLOCK_SIZE, d_k)
                V_j = V[:, kv_start:kv_end, :]  # (batch, BLOCK_SIZE, d_v)

                # compute attention scores for this tile
                # S_ij: (batch, BLOCK_SIZE_q, BLOCK_SIZE_kv)
                S_ij = torch.einsum('bid,bjd->bij', Q_i, K_j) * scale

                # local max for numerical stability
                m_ij = S_ij.max(dim=-1).values  # (batch, BLOCK_SIZE_q)

                # update running max
                # note this is elementwise
                m_i_new = torch.maximum(m_i, m_ij)  # (batch, BLOCK_SIZE_q)

                # compute exponentials with stability correction
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # (batch, BLOCK_SIZE_q, BLOCK_SIZE_kv)

                # correction factor for previous running sum
                correction = torch.exp(m_i - m_i_new)  # (batch, BLOCK_SIZE_q)

                # update running sum
                d_i = correction * d_i + P_ij.sum(dim=-1)  # (batch, BLOCK_SIZE_q)

                # update output accumulator
                # note "correction" is NOT doing normalisation (this happens after inner loop)
                # it is simply correcting the effect of the old values of m_i
                O_i = (correction.unsqueeze(-1) * O_i.float() + 
                       torch.einsum('bij,bjd->bid', P_ij.float(), V_j.float()))

                # update running max
                m_i = m_i_new

            # normalise output for this query tile
            O_i = O_i / d_i.unsqueeze(-1)
            O[:, q_start:q_end, :] = O_i.to(Q.dtype)

            # compute logsumexp: L = m + log(d)
            L[:, q_start:q_end] = m_i + torch.log(d_i)

        # save for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors  # note: save these in forward!
        is_causal = ctx.is_causal
        dQ, dK, dV = flash_attention_backward_compiled(
            Q, K, V, O, grad_output.contiguous(), L, is_causal
        )
        return dQ, dK, dV, None

def flash_attention_pytorch(Q, K, V, is_causal=False):
    return FlashAttentionPyTorch.apply(Q, K, V, is_causal)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,  # <-- new constexpr flag
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # query indices for this tile
    query_start = query_tile_index * Q_TILE_SIZE
    query_indices = query_start + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE_SIZE,)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q = tl.load(Q_block_ptr)

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    d_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(n_key_tiles):
        # key indices for this tile
        key_start = j * K_TILE_SIZE
        key_indices = key_start + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE,)

        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)

        S_ij = tl.dot(Q, tl.trans(K)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # apply causal mask
        if IS_CAUSAL:
            # mask[i, j] = True if query_i can attend to key_j (j <= i)
            causal_mask = query_indices[:, None] >= key_indices[None, :]  # (Q_TILE_SIZE, K_TILE_SIZE)
            S_ij = tl.where(causal_mask, S_ij, float('-inf'))

        m_ij = tl.max(S_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        correction = tl.exp(m_i - m_i_new)
        P_ij = tl.exp(S_ij - m_i_new[:, None])
        d_i = correction * d_i + tl.sum(P_ij, axis=1)
        O_i = correction[:, None] * O_i
        O_i = tl.dot(P_ij.to(V_block_ptr.type.element_ty), V, acc=O_i)
        m_i = m_i_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / d_i[:, None]
    L_i = m_i + tl.log(d_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i)

class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        FlashAttention-2 forward pass using Triton kernel.

        Args:
            Q: (batch, seq_len, d_k)
            K: (batch, seq_len, d_k)
            V: (batch, seq_len, d_v)
            is_causal: bool (ignored for now)

        Returns:
            O: (batch, seq_len, d_v)
        """
        batch, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        scale = 1.0 / math.sqrt(D)

        # tile sizes — powers of 2, at least 16
        Q_TILE_SIZE = max(16, triton.next_power_of_2(min(N_QUERIES, 64)))
        K_TILE_SIZE = max(16, triton.next_power_of_2(min(N_KEYS, 64)))

        # ensure contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        # allocate output tensors
        O = torch.empty(batch, N_QUERIES, D, device=Q.device, dtype=Q.dtype)
        L = torch.empty(batch, N_QUERIES, device=Q.device, dtype=torch.float32)

        # launch grid: (num_query_tiles, batch_size)
        grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=is_causal,  # <-- pass flag
        )

        # save for backward
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors  # match forward save order: L, Q, K, V, O
        is_causal = ctx.is_causal

        # use compiled pytorch backward
        dQ, dK, dV = flash_attention_backward_compiled(
            Q, K, V, O, dO.contiguous(), L, is_causal
        )

        # return gradients for each forward input
        # is_causal has no gradient (not a tensor)
        return dQ, dK, dV, None

def get_flash_autograd_function_triton():
    return FlashAttentionTriton

def flash_attention_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    orig_dtype = Q.dtype
    # work entirely in float32 for numerical stability
    Q  = Q.float()
    K  = K.float()
    V  = V.float()
    O  = O.float()
    dO = dO.float()
    # L is already float32

    # handle arbitrary batch/head dimensions using ...
    scale = 1.0 / math.sqrt(Q.shape[-1])
    seq_len = Q.shape[-2]

    # D vector: (..., seq_len)
    D = (O * dO).sum(dim=-1)

    # recompute scores: (..., seq_q, seq_k)
    S = torch.einsum('...id,...jd->...ij', Q, K) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device), diagonal=1
        ).bool()
        S = S.masked_fill(mask, float('-inf'))

    # recompute P from saved L
    P = torch.exp(S - L.unsqueeze(-1))

    # dV: (..., seq_k, d_v)
    dV = torch.einsum('...ij,...id->...jd', P, dO)

    # dP: (..., seq_q, seq_k)
    dP = torch.einsum('...id,...jd->...ij', dO, V)

    # dS: (..., seq_q, seq_k)
    dS = P * (dP - D.unsqueeze(-1))

    if is_causal:
        dS = dS.masked_fill(mask, 0.0)

    # dQ: (..., seq_q, d_k)
    dQ = torch.einsum('...ij,...jd->...id', dS, K) * scale

    # dK: (..., seq_k, d_k)
    dK = torch.einsum('...ij,...id->...jd', dS, Q) * scale

    # cast back to original dtype
    return dQ.to(orig_dtype), dK.to(orig_dtype), dV.to(orig_dtype)

# compile for performance
flash_attention_backward_compiled = torch.compile(flash_attention_backward_pytorch)