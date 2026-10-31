"""LFM3: the family, its params for pipeline/lfm3_model.py, the converter, and its names.

tests/fixtures/lfm3-1.7b.config.json is LiquidAI/LFM3-1.7B's config.json at ac2586a3. The
model definition itself is checked against the checkpoint's modeling_lfm3.py on real
weights, which needs the 3.4 GB download and the remote code, so it is not a unit test:
docs/PLAN.md has the numbers (identical fp32 logits without a cache; 1.8e-4 with the custom
KV-cache ops, chunked prefill over the ring and 40 decode steps).
"""

from __future__ import annotations

import pytest
from conftest import load_json, make_source

from pipeline import convert, eligibility, families, naming, settings

LFM3_PARAMS = 1_728_801_536


def lfm3_config(**overrides) -> dict:
    config = load_json("lfm3-1.7b.config.json")
    config.update(overrides)
    return config


def test_lfm3_is_its_own_family_with_xnnpack_only():
    family = families.family_for(lfm3_config())
    assert family.key == "lfm3"
    assert family.supports("xnnpack")
    for backend in ("vulkan", "qnn", "mtk"):
        assert not family.supports(backend)


def test_lfm3_plan_carries_both_ropes_the_window_and_the_convolution_width():
    plan = families.xnnpack_plan(families.family_for(lfm3_config()), lfm3_config())
    assert plan.converter == "lfm3" and plan.lfm3 and plan.state_reset_at_zero
    p = plan.params
    # Global layers: theta 1e6 on a zero-tail half of the rotary frequencies.
    assert p["rope_theta"] == 1_000_000.0 and p["partial_rotary_factor"] == 0.5
    # Windowed layers: theta 1e4, fully rotated.
    assert p["local_rope_theta"] == 10_000.0
    assert p["sliding_window"] == 256
    assert p["attention_kwargs"] == {"conv_L_cache": 7}
    assert p["hidden_dim"] == 7680 and p["head_dim"] == 128 and p["n_kv_heads"] == 4
    assert p["layer_types"].count("conv") == 13
    assert p["layer_types"].count("full_attention") == 6
    assert p["layer_types"].count("sliding_attention") == 5
    assert p["post_attention_norm"] and p["post_ffn_norm"]


def test_lfm3_attends_on_eleven_of_twenty_four_layers():
    assert families.attending_layers(lfm3_config()) == 11


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"conv_bias": True}, "convolution bias"),
        ({"use_sandwich_norm": False}, "sandwich"),
        ({"tie_word_embeddings": False}, "untied"),
        ({"sliding_window": None}, "sliding_window"),
        (
            {
                "rope_parameters": {
                    "full_attention": {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                }
            },
            "RoPE",
        ),
    ],
)
def test_lfm3_refuses_what_the_definition_does_not_implement(overrides, reason):
    with pytest.raises(families.UnsupportedModel, match=reason):
        families.xnnpack_plan(families.family_for(lfm3_config()), lfm3_config(**overrides))


# Every tensor name in LFM3-1.7B's model.safetensors (layer 0 is a convolution, layer 2
# global attention, layer 4 windowed attention).
LFM3_KEYS = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.embedding_norm.weight": "norm.weight",
    "model.layers.0.conv.conv.weight": "layers.0.conv.conv.weight",
    "model.layers.0.conv.in_proj.weight": "layers.0.conv.in_proj.weight",
    "model.layers.0.conv.out_proj.weight": "layers.0.conv.out_proj.weight",
    "model.layers.0.feed_forward.w1.weight": "layers.0.feed_forward.w1.weight",
    "model.layers.0.feed_forward.w2.weight": "layers.0.feed_forward.w2.weight",
    "model.layers.0.feed_forward.w3.weight": "layers.0.feed_forward.w3.weight",
    "model.layers.0.ffn_norm.weight": "layers.0.ffn_norm.weight",
    "model.layers.0.operator_norm.weight": "layers.0.attention_norm.weight",
    "model.layers.0.post_ffn_norm.weight": "layers.0.post_ffn_norm.weight",
    "model.layers.0.post_operator_norm.weight": "layers.0.post_attention_norm.weight",
    "model.layers.2.self_attn.gate_proj.weight": "layers.2.attention.gate_proj.weight",
    "model.layers.2.self_attn.k_norm.weight": "layers.2.attention.k_norm_fn.weight",
    "model.layers.2.self_attn.k_proj.weight": "layers.2.attention.wk.weight",
    "model.layers.2.self_attn.o_proj.weight": "layers.2.attention.wo.weight",
    "model.layers.2.self_attn.q_norm.weight": "layers.2.attention.q_norm_fn.weight",
    "model.layers.2.self_attn.q_proj.weight": "layers.2.attention.wq.weight",
    "model.layers.4.self_attn.v_proj.weight": "layers.4.attention.wv.weight",
}


def test_every_lfm3_tensor_has_an_executorch_name():
    for hf, et in LFM3_KEYS.items():
        assert convert.map_key(hf, "lfm3") == et, hf


def test_lfm3_does_not_borrow_lfm2s_attention_names():
    # LFM2 says out_proj and q_layernorm; an LFM3 checkpoint carrying them is not LFM3.
    for key in ("model.layers.2.self_attn.out_proj.weight", "model.layers.2.self_attn.q_layernorm.weight"):
        with pytest.raises(KeyError):
            convert.map_key(key, "lfm3")


def test_lfm3_splits_the_fused_convolution_projection_like_lfm2():
    assert convert.SPLIT_THREE["lfm3"] == convert.SPLIT_THREE["lfm2"]


def test_lfm3_is_exported_ahead_of_the_apps_template():
    name = naming.app_model_name("experimentalmachines/LFM3-1.7B-ExecuTorch", "xnnpack/LFM3-1.7B-8da4w-gptq-2k.pte")
    # The Kotlin port stays the app's: no LFM3 template there yet.
    assert naming.app_family(name) is None
    assert naming.pipeline_family(name) == "lfm3"
    assert naming.ahead_of_app("LFM3-1.7B")
    assert not naming.ahead_of_app("LFM2.5-1.2B-Instruct")
    repo, path = "experimentalmachines/LFM3-1.7B-ExecuTorch", "xnnpack/LFM3-1.7B-8da4w-gptq-32k.pte"
    assert naming.check_app_rules(repo, path, "xnnpack") == []


def test_the_vision_model_stays_out():
    assert naming.pipeline_family("LFM3-VL-2B-ExecuTorch") is None


def test_lfm3_chat_model_is_eligible_for_xnnpack_only():
    cfg = settings.load()
    verdict = eligibility.evaluate(make_source("LiquidAI/LFM3-1.7B", lfm3_config(), total_params=LFM3_PARAMS), cfg)
    assert verdict.reasons == []
    assert verdict.variant == "instruct"
    assert verdict.backends["xnnpack"] is None
    assert all(verdict.backends[b] for b in ("vulkan", "qnn", "mtk"))


def test_lfm3_base_is_a_base_checkpoint():
    cfg = settings.load()
    verdict = eligibility.evaluate(make_source("LiquidAI/LFM3-1.7B-Base", lfm3_config(), total_params=LFM3_PARAMS), cfg)
    assert any("base checkpoint" in r for r in verdict.reasons)


def test_rope_tables_leave_the_tail_unrotated():
    torch = pytest.importorskip("torch")
    pytest.importorskip("executorch")
    from pipeline import lfm3_model

    cos, sin = lfm3_model.rope_tables(128, 1e6, 0.5, 8)
    assert cos.shape == (8, 64)
    # 32 rotated frequencies from the full 128-dim progression, then 32 at zero.
    assert torch.all(cos[:, 32:] == 1) and torch.all(sin[:, 32:] == 0)
    assert torch.isclose(sin[1, 1], torch.sin(torch.tensor(1e6 ** (-2 / 128))))


def test_remote_code_runs_only_for_a_trusted_publisher_and_local_modules():
    from pipeline import hub

    official = make_source("LiquidAI/LFM3-1.7B", lfm3_config(), total_params=LFM3_PARAMS)
    assert hub.remote_code_allowed(official)
    finetune = make_source("someone/LFM3-1.7B-finetune", lfm3_config(), total_params=LFM3_PARAMS)
    assert not hub.remote_code_allowed(finetune)
    elsewhere = lfm3_config(auto_map={"AutoModelForCausalLM": "other/repo--modeling_lfm3.Lfm3ForCausalLM"})
    assert not hub.remote_code_allowed(make_source("LiquidAI/LFM3-1.7B", elsewhere, total_params=LFM3_PARAMS))
