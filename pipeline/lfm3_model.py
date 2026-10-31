"""LFM3 for ExecuTorch's ``export_llm``: the model definition ExecuTorch 1.5.1 does not have.

LiquidAI's LFM3-1.7B (2026-10-23) is LFM2's hybrid grown three ways, none of which the
1.5.1 transformer can express (examples/models/llama/llama_transformer.py, model_args.py):

* A third layer type. Of 24 layers, 13 are gated short convolutions, 6 attend globally and
  5 attend over a sliding window of 256 tokens. ``construct_transformer`` knows ``conv`` and
  treats anything else as global attention, so the windowed layers would see everything.
* Two RoPEs. Global layers rotate a quarter of each head at theta 1e6 ("zero-tail" partial
  RoPE: the first ``partial_rotary_factor * head_dim / 2`` frequencies follow the full head
  dimension's progression and the rest are zero), windowed layers rotate all of it at theta
  1e4. ExecuTorch computes one table for the whole model.
* Per-layer extras: a sigmoid output gate per head (``gate_proj``, hidden -> n_heads; 1.5.1's
  ``use_attn_o_gate`` is per element), RMSNorm after each operator and after each feed-forward
  ("sandwich" norms; 1.5.1 declares ``post_attention_norm`` and never reads it), and a
  width-7 convolution (``ShortConvBlock`` hard-codes 3).

Everything else is LFM2's, and the parts are ExecuTorch's own where they exist: RMSNorm,
FeedForward, the KV cache and SDPA modules that ``export_llm`` swaps for its custom ops, the
ring-buffer bookkeeping it uses for Gemma 3. Reference: modeling_lfm3.py in the checkpoint
(``trust_remote_code``; transformers has no LFM3 yet).

The windowed layers keep their keys in a ring, not a full-length cache: ``window + prefill
chunk`` slots, which is what a chunk of up to ``max_seq_len`` tokens needs to still see the
256 before it (ExecuTorch's own ``RingKVCache`` has ``2 * window`` and asserts the chunk fits
in it, which a 2048-token prefill does not). At 32k that is 2,304 slots instead of 32,768 for
5 of the 11 attending layers, and no window-squared mask anywhere.

The short convolution's state is cleared in the graph at position zero, as for LFM2
(pipeline/lfm2_state.py, and the export says so with ``get_state_reset_at_zero``).

``apply()`` makes ``construct_transformer`` build this model when the params say LFM3 (a
``sliding_attention`` layer type), and leaves every other model to upstream. It must run in
any process that builds the model: ``pipeline.export_with_codes --lfm3`` and the solve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

LAYER_TYPES = ("conv", "full_attention", "sliding_attention")


def is_lfm3(args) -> bool:
    return bool(args.layer_types) and "sliding_attention" in args.layer_types


def rope_tables(head_dim: int, theta: float, partial: float, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin, [length, head_dim / 2], as Lfm3RotaryEmbedding computes them.

    The zero-tail layout (compute_default_rope_parameters): ``partial * head_dim // 2``
    frequencies over the full ``head_dim``, then zeros, so those dimensions get cos 1, sin 0.
    """
    # device="cpu" throughout: Llama2Model builds the module under torch.device("meta"), and
    # a non-persistent buffer is never loaded from the checkpoint, so a meta one stays empty.
    angles = int(partial * head_dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float, device="cpu") / head_dim))
    inv = torch.cat((inv, torch.zeros(head_dim // 2 - angles, dtype=torch.float, device="cpu")))
    freqs = torch.outer(torch.arange(length, dtype=torch.float, device="cpu"), inv)
    return freqs.cos(), freqs.sin()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF's rotate_half RoPE on ``x`` [B, S, H, D] with cos/sin [S, D/2]."""
    cos = torch.cat((cos, cos), -1)[None, :, None, :]
    sin = torch.cat((sin, sin), -1)[None, :, None, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * cos + rotated * sin


class RingCache(nn.Module):
    """K and V of one windowed layer in ``window + chunk`` slots, on ExecuTorch's custom ops.

    The bookkeeping is ExecuTorch's (attention.py CachePositionsManager and
    _create_causal_mask_for_ring_buffer, as CustomRingKVCache uses them); only the size
    differs. A position-zero call resets every slot to empty, so a new prompt never sees the
    last one. Not a KVCache or CustomKVCache subclass on purpose: ``export_llm``'s
    replace_kv_cache_with_custom_kv_cache would otherwise swap it for a plain cache.
    """

    def __init__(self, n_kv_heads: int, head_dim: int, window: int, chunk: int):
        super().__init__()
        from executorch.examples.models.llama.attention import CachePositionsManager
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401  (llama:: ops)

        self.window = window
        self.capacity = window + chunk
        shape = (1, self.capacity, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=torch.float32, device="cpu"))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=torch.float32, device="cpu"))
        self.positions = CachePositionsManager(self.capacity)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """q [B, S, H, D], k and v [B, S, KV, D] -> [B, S, H, D]."""
        from executorch.examples.models.llama.attention import _create_causal_mask_for_ring_buffer

        seqlen = q.shape[1]
        start = input_pos[0].item()
        torch._check_is_size(start)
        indices = self.positions.calculate_positions_and_update_indices(input_pos, seqlen).unsqueeze(0)
        torch.ops.llama.update_cache_with_indices(k, self.k_cache, start, indices)
        torch.ops.llama.update_cache_with_indices(v, self.v_cache, start, indices)
        mask = _create_causal_mask_for_ring_buffer(self.positions.cache_positions, self.window, start, seqlen)
        return torch.ops.llama.custom_sdpa(q, self.k_cache, self.v_cache, 0, mask, 0, False)


class Lfm3Attention(nn.Module):
    """Lfm3Attention: QK RMSNorm before RoPE, a sigmoid gate per head, a window or not."""

    def __init__(self, args, layer_id: int, sliding: bool):
        super().__init__()
        from executorch.examples.models.llama.attention import SDPA, KVCache
        from executorch.examples.models.llama.norm import RMSNorm

        self.layer_id = layer_id
        self.sliding = sliding
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.dim = args.dim
        self.window = int(args.sliding_window)
        self.max_context_len = args.max_context_len
        self.use_kv_cache = args.use_kv_cache
        self.enable_dynamic_shape = args.enable_dynamic_shape
        q_dim, kv_dim = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        self.wq = nn.Linear(args.dim, q_dim, bias=False)
        self.wk = nn.Linear(args.dim, kv_dim, bias=False)
        self.wv = nn.Linear(args.dim, kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, args.dim, bias=False)
        self.gate_proj = nn.Linear(args.dim, self.n_heads, bias=False)
        self.q_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)
        # Which of Rope's two tables this layer rotates with.
        self.rope_index = Rope.LOCAL if sliding else Rope.GLOBAL
        if self.use_kv_cache:
            if sliding:
                self.ring = RingCache(self.n_kv_heads, self.head_dim, self.window, args.max_seq_len)
            else:
                # ExecuTorch's own modules, so export_llm swaps them for CustomKVCache and
                # SDPACustom (causal from start_pos) exactly as for every other model.
                self.kv_cache = KVCache(
                    args.max_batch_size,
                    self.max_context_len,
                    self.n_kv_heads,
                    self.head_dim,
                    args.enable_dynamic_shape,
                )
                self.SDPA = SDPA(q_dim, self.head_dim, self.n_rep, self.max_context_len)

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor | None, freqs_cos, freqs_sin) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_norm_fn(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim))
        k = self.k_norm_fn(self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim))
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        cos, sin = freqs_cos[self.rope_index], freqs_sin[self.rope_index]
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)

        if not self.use_kv_cache:
            out = self._attend_whole(q, k, v)
        elif self.sliding:
            out = self.ring.attend(q, k, v, input_pos)
        else:
            out = self._attend_cached(q, k, v, input_pos, bsz, seqlen)

        gate = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        out = (out.reshape(bsz, seqlen, self.n_heads, self.head_dim) * gate).reshape(bsz, seqlen, -1)
        return self.wo(out)

    def _attend_cached(self, q, k, v, input_pos, bsz, seqlen):
        from executorch.examples.models.llama.attention import SDPA

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k, v = self.kv_cache.update(input_pos, k, v)
        mask = None
        if isinstance(self.SDPA, SDPA):
            # Only the eager module reads a mask; SDPACustom is causal from start_pos. Built
            # here rather than kept as a context-squared buffer per layer.
            start = input_pos[0].item()
            rows = start + torch.arange(seqlen, dtype=torch.long).view(-1, 1)
            mask = torch.arange(self.max_context_len, dtype=torch.long).view(1, -1) <= rows
        return self.SDPA(input_pos, q, k, v, bsz, seqlen, mask)

    def _attend_whole(self, q, k, v):
        """No cache (the solve): the whole sequence at once, causal and maybe windowed."""
        seqlen = q.shape[1]
        i = torch.arange(seqlen).view(-1, 1)
        j = torch.arange(seqlen).view(1, -1)
        mask = j <= i
        if self.sliding:
            mask = mask & (j > i - self.window)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2)


class ShortConv(nn.Module):
    """LFM2's gated short convolution, any width, its state cleared at position zero."""

    def __init__(self, dim: int, width: int):
        super().__init__()
        self.width = width
        self.B_proj = nn.Linear(dim, dim, bias=False)
        self.C_proj = nn.Linear(dim, dim, bias=False)
        self.x_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, kernel_size=width, padding=0, groups=dim, bias=False)
        self.register_buffer("conv_state", torch.zeros(1, dim, width - 1, device="cpu"))

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor | None) -> torch.Tensor:
        B = self.B_proj(x).transpose(-1, -2)
        C = self.C_proj(x).transpose(-1, -2)
        Bx = B * self.x_proj(x).transpose(-1, -2)
        if input_pos is None:
            state = torch.zeros_like(self.conv_state)
        else:
            # min(position, 1): zeros for a prompt that starts at 0, the carried state for a
            # continuation (pipeline/lfm2_state.py has why it is a clamp, not a comparison).
            keep = torch.clamp(input_pos.reshape(-1)[:1], max=1).to(self.conv_state.dtype).view(1, 1, 1)
            state = self.conv_state * keep
        Bx = torch.cat([state, Bx], dim=-1)
        with torch.no_grad():
            self.conv_state.copy_(Bx[..., -(self.width - 1) :])
        y = C * self.conv(Bx)[..., : x.size(1)]
        return self.out_proj(y.transpose(-1, -2).contiguous())


class Block(nn.Module):
    """Lfm3DecoderLayer: operator (convolution or attention), feed-forward, sandwich norms."""

    def __init__(self, args, layer_id: int, layer_type: str, conv_width: int, sandwich: bool):
        super().__init__()
        from executorch.examples.models.llama.feed_forward import FeedForward
        from executorch.examples.models.llama.norm import RMSNorm

        if layer_type == "conv":
            self.conv = ShortConv(args.dim, conv_width)
        else:
            self.attention = Lfm3Attention(args, layer_id, sliding=layer_type == "sliding_attention")
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim)
        # attention_norm / ffn_norm: the names ExecuTorch's blocks use (operator_norm in HF).
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.sandwich = sandwich
        if sandwich:
            self.post_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.post_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos=None, freqs_sin=None, attn_options=None):
        input_pos = attn_options.get("input_pos") if attn_options else None
        if hasattr(self, "conv"):
            h = self.conv(self.attention_norm(x), input_pos)
        else:
            h = self.attention(self.attention_norm(x), input_pos, freqs_cos, freqs_sin)
        if self.sandwich:
            h = self.post_attention_norm(h)
        h = x + h
        ffn = self.feed_forward(self.ffn_norm(h))
        if self.sandwich:
            ffn = self.post_ffn_norm(ffn)
        return h + ffn, None


class Rope(nn.Module):
    """Both RoPEs in one table, [2, context, head_dim / 2]: index GLOBAL or LOCAL.

    In the Transformer's RoPE slot, so the tables are stored once and sliced once per call
    (Transformer.forward hands the same freqs to every layer, which picks its own row) rather
    than once per attending layer: 11 copies would add ~176 MB to a 32k file.
    """

    GLOBAL, LOCAL = 0, 1

    def __init__(self, args):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.max_context_len = args.max_context_len
        tables = (
            rope_tables(args.head_dim, float(args.rope_theta), float(args.partial_rotary_factor), args.max_context_len),
            rope_tables(args.head_dim, float(args.local_rope_theta), 1.0, args.max_context_len),
        )
        self.register_buffer("freqs_cos", torch.stack([t[0] for t in tables]), persistent=False)
        self.register_buffer("freqs_sin", torch.stack([t[1] for t in tables]), persistent=False)

    def get_freqs(self, input_pos, seq_len: int):
        if not self.use_kv_cache or input_pos is None:
            return self.freqs_cos[:, :seq_len], self.freqs_sin[:, :seq_len]
        # As upstream's Rope.get_freqs with a dynamic shape.
        start = input_pos[-1].item()
        torch._check_is_size(start)
        torch._check(start < self.max_context_len)
        return self.freqs_cos.narrow(1, start, seq_len), self.freqs_sin.narrow(1, start, seq_len)


def construct(args):
    from executorch.examples.models.llama.llama_transformer import Transformer

    unknown = set(args.layer_types) - set(LAYER_TYPES)
    if unknown:
        raise ValueError(f"LFM3 layer types {sorted(unknown)} are not implemented")
    if len(args.layer_types) != args.n_layers:
        raise ValueError(f"{len(args.layer_types)} layer types for {args.n_layers} layers")
    extra = args.attention_kwargs or {}
    conv_width = int(extra.get("conv_L_cache", 3))
    sandwich = bool(args.post_attention_norm and args.post_ffn_norm)
    layers = nn.ModuleList(
        Block(args, i, layer_type, conv_width, sandwich) for i, layer_type in enumerate(args.layer_types)
    )
    return Transformer(args, layers, Rope(args))


_APPLIED = False


def apply() -> None:
    """Route ``construct_transformer`` to LFM3 for LFM3 params. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    from executorch.examples.models.llama import llama_transformer
    from executorch.examples.models.llama import model as llama_model

    upstream = llama_transformer.construct_transformer

    def construct_transformer(model_args):
        return construct(model_args) if is_lfm3(model_args) else upstream(model_args)

    llama_transformer.construct_transformer = construct_transformer
    # model.py imported the name, so its binding is the one Llama2Model calls.
    llama_model.construct_transformer = construct_transformer
    _APPLIED = True
