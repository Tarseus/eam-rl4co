from __future__ import annotations

import json
import logging
import os
import re
from hashlib import sha1
from typing import Any, Mapping

from .free_loss_compiler import parse_free_loss_from_text
from .free_loss_ir import FreeLossIR, FreeLossImplementationHint
from .free_loss_llm_ops import _call_llm, _extract_json_object, configure_llm_run, llm_cache_stats
from .pref_builder_ir import PreferenceBuilderIR, PreferenceBuilderImplementationHint
from .pref_builder_llm_ops import _append_global_feedback, _append_prompt_context_block, _parse_pref_builder_from_text


LOGGER = logging.getLogger(__name__)

_BUILDER_WEIGHT_FAMILIES = {
    "gap_linear",
    "gap_square",
    "gap_rank_blend",
    "gap_regret_blend",
    "margin_rank_blend",
    "margin_regret_blend",
    "gap_bandpass",
}
_LOSS_TEMPLATES = {
    "weighted_softplus_gap",
    "weighted_logsigmoid_gap",
    "weighted_softplus_rank",
}


def _read_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        LOGGER.warning("Prompt file missing (%s); using built-in fallback prompt.", path)
        return _fallback_prompt()


def _fallback_prompt() -> str:
    return """You are generating one bound preference-builder and preference-loss pair.

Return ONLY one JSON object. Do not output code, markdown, alternatives, or explanations.

Schema:
{
  "builder": {
    "name": "...",
    "template": "all_pairs_reweight",
    "weight_family": "gap_linear",
    "intuition": "...",
    "hyperparams": {"weight_scale": 1.0}
  },
  "loss": {
    "name": "...",
    "template": "weighted_softplus_gap",
    "intuition": "...",
    "pseudocode": "...",
    "hyperparams": {"temperature": 1.0, "gap_scale": 0.5, "target_margin": 0.0}
  }
}

Allowed builder.weight_family values: gap_linear, gap_square, gap_rank_blend,
gap_regret_blend, margin_rank_blend, margin_regret_blend, gap_bandpass.
Allowed loss.template values: weighted_softplus_gap, weighted_logsigmoid_gap,
weighted_softplus_rank.
The runtime expands templates into executable code and evaluates only this bound pair.
"""


def _extract_mapping_field(obj: Mapping[str, Any], names: tuple[str, ...]) -> Mapping[str, Any]:
    for name in names:
        value = obj.get(name)
        if isinstance(value, Mapping):
            return value
    raise ValueError(f"Joint pair JSON missing object field: one of {list(names)}")


def _safe_name(value: Any, fallback: str) -> str:
    name = str(value or "").strip() or str(fallback)
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    return name[:80] or str(fallback)


def _mapping_hyperparams(obj: Mapping[str, Any]) -> dict[str, Any]:
    raw = obj.get("hyperparams", {}) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _normalize_template_constraints(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    constraints = dict(raw or {}) if isinstance(raw, Mapping) else {}
    out: dict[str, Any] = {"force_template": bool(constraints.get("force_template", True))}
    weight_family = str(
        constraints.get("target_weight_family")
        or constraints.get("weight_family")
        or ""
    ).strip().lower()
    if weight_family in _BUILDER_WEIGHT_FAMILIES:
        out["target_weight_family"] = weight_family
    loss_template = str(
        constraints.get("target_loss_template")
        or constraints.get("loss_template")
        or ""
    ).strip().lower()
    if loss_template in _LOSS_TEMPLATES:
        out["target_loss_template"] = loss_template
    return out


def _with_template_constraints(
    obj: Mapping[str, Any],
    *,
    target_weight_family: str | None = None,
    target_loss_template: str | None = None,
) -> dict[str, Any]:
    out = dict(obj)
    hyper = _mapping_hyperparams(out)
    if target_weight_family:
        out["weight_family"] = str(target_weight_family)
        hyper["weight_family"] = str(target_weight_family)
    if target_loss_template:
        out["template"] = str(target_loss_template)
        hyper["template"] = str(target_loss_template)
    out["hyperparams"] = hyper
    return out


def _safe_float(value: Any, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if out != out or out in {float("inf"), float("-inf")}:
        out = float(default)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return float(out)


def _builder_ir_from_template(obj: Mapping[str, Any]) -> PreferenceBuilderIR:
    hyper = _mapping_hyperparams(obj)
    weight_family = str(obj.get("weight_family") or hyper.get("weight_family") or "gap_linear").strip().lower()
    if weight_family not in _BUILDER_WEIGHT_FAMILIES:
        weight_family = "gap_linear"
    weight_scale = _safe_float(hyper.get("weight_scale", obj.get("weight_scale", 1.0)), 1.0, lo=0.05, hi=8.0)
    clamp_lo = _safe_float(hyper.get("clamp_lo", 0.25), 0.25, lo=0.0, hi=4.0)
    clamp_hi = _safe_float(hyper.get("clamp_hi", 4.0), 4.0, lo=max(0.05, clamp_lo), hi=16.0)
    lambda_rank = _safe_float(hyper.get("lambda_rank", 1.0), 1.0, lo=0.0, hi=8.0)
    lambda_regret = _safe_float(hyper.get("lambda_regret", 1.0), 1.0, lo=0.0, hi=8.0)
    gamma = _safe_float(hyper.get("gamma", 0.5), 0.5, lo=0.0, hi=2.0)
    center = _safe_float(hyper.get("center", 0.5), 0.5, lo=0.0, hi=1.0)
    beta_band = _safe_float(hyper.get("beta_band", 4.0), 4.0, lo=0.1, hi=16.0)

    if weight_family == "gap_linear":
        raw_expr = "gap_scaled.clamp_min(0.0)"
        extra_lines: list[str] = []
    elif weight_family == "gap_square":
        raw_expr = "gap_scaled.clamp_min(0.0).pow(2)"
        extra_lines = []
    elif weight_family == "gap_rank_blend":
        raw_expr = f"gap_scaled.clamp_min(0.0).sqrt() * (1.0 + {lambda_rank:.8g} * rank_span)"
        extra_lines = ["    rank_span = rank[b_idx, loser_idx] - rank[b_idx, winner_idx]"]
    elif weight_family == "gap_regret_blend":
        raw_expr = f"gap_scaled.clamp_min(0.0).sqrt() * (1.0 + {lambda_regret:.8g} * regret_span)"
        extra_lines = [
            "    regret_span = regret[b_idx, loser_idx] - regret[b_idx, winner_idx]",
            "    regret_span = regret_span / (1.0 + instance_regret_scale[b_idx].clamp_min(eps))",
        ]
    elif weight_family == "margin_rank_blend":
        raw_expr = f"torch.sigmoid(4.0 * rank_span) / (margin_scaled + eps).pow({gamma:.8g})"
        extra_lines = [
            "    rank_span = rank[b_idx, loser_idx] - rank[b_idx, winner_idx]",
            "    margin_abs = (log_prob[b_idx, winner_idx] - log_prob[b_idx, loser_idx]).abs()",
            "    margin_scaled = margin_abs / instance_log_prob_scale[b_idx].clamp_min(eps)",
        ]
    elif weight_family == "margin_regret_blend":
        raw_expr = f"(1.0 + {lambda_regret:.8g} * regret_span) / (margin_scaled + eps).pow({gamma:.8g})"
        extra_lines = [
            "    regret_span = regret[b_idx, loser_idx] - regret[b_idx, winner_idx]",
            "    regret_span = regret_span / (1.0 + instance_regret_scale[b_idx].clamp_min(eps))",
            "    margin_abs = (log_prob[b_idx, winner_idx] - log_prob[b_idx, loser_idx]).abs()",
            "    margin_scaled = margin_abs / instance_log_prob_scale[b_idx].clamp_min(eps)",
        ]
    else:
        raw_expr = f"torch.exp(-{beta_band:.8g} * (gap_unit - {center:.8g}).abs())"
        extra_lines = [
            "    gap_min = torch.zeros(batch_size, dtype=objective.dtype, device=objective.device)",
            "    gap_max = torch.zeros(batch_size, dtype=objective.dtype, device=objective.device)",
            "    gap_min.scatter_reduce_(0, b_idx, gap_scaled, reduce='amin', include_self=False)",
            "    gap_max.scatter_reduce_(0, b_idx, gap_scaled, reduce='amax', include_self=False)",
            "    gap_unit = (gap_scaled - gap_min[b_idx]) / (gap_max[b_idx] - gap_min[b_idx] + eps)",
        ]

    code_lines = [
        "def generated_builder(feature_cache, extra):",
        "    objective = feature_cache['objective']",
        "    log_prob = feature_cache['log_prob']",
        "    rank = feature_cache['rank']",
        "    regret = feature_cache['regret']",
        "    batch_size = int(objective.shape[0])",
        "    mask = objective[:, :, None] < objective[:, None, :]",
        "    b_idx, winner_idx, loser_idx = mask.nonzero(as_tuple=True)",
        "    if int(b_idx.numel()) <= 0:",
        f"        return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=None, meta={{'builder': 'all_pairs_reweight_template', 'weight_family': '{weight_family}'}})",
        "    eps = 1e-6",
        "    gap = objective[b_idx, loser_idx] - objective[b_idx, winner_idx]",
        "    instance_obj_scale = feature_cache.get('instance_obj_mad')",
        "    if instance_obj_scale is None:",
        "        instance_obj_scale = feature_cache.get('instance_obj_std')",
        "    if instance_obj_scale is None:",
        "        instance_obj_scale = torch.ones((batch_size,), dtype=objective.dtype, device=objective.device)",
        "    instance_log_prob_scale = feature_cache.get('instance_log_prob_std')",
        "    if instance_log_prob_scale is None:",
        "        instance_log_prob_scale = torch.ones((batch_size,), dtype=objective.dtype, device=objective.device)",
        "    instance_regret_scale = feature_cache.get('instance_regret_std')",
        "    if instance_regret_scale is None:",
        "        instance_regret_scale = torch.ones((batch_size,), dtype=objective.dtype, device=objective.device)",
        "    gap_scaled = gap / instance_obj_scale[b_idx].clamp_min(eps)",
        *extra_lines,
        f"    raw = ({raw_expr}) * {weight_scale:.8g}",
        "    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)",
        f"    weight = raw.clamp({clamp_lo:.8g}, {clamp_hi:.8g})",
        f"    return PrefBatch(mode='pairwise', pair_idx=(b_idx, winner_idx, loser_idx), weight=weight, meta={{'builder': 'all_pairs_reweight_template', 'weight_family': '{weight_family}'}})",
    ]
    name = _safe_name(obj.get("name"), f"builder_template_{weight_family}")
    hyper_out = dict(hyper)
    hyper_out.update(
        {
            "geometry_family": "dense_all_pairs",
            "cap_family": "uncapped_full",
            "weight_family": str(weight_family),
            "constraint_family": "clamp_only",
            "template": "all_pairs_reweight",
        }
    )
    return PreferenceBuilderIR(
        name=name,
        intuition=str(obj.get("intuition") or f"template all-pairs reweight builder using {weight_family}"),
        implementation_hint=PreferenceBuilderImplementationHint(
            expects=[
                "objective",
                "log_prob",
                "rank",
                "regret",
                "instance_obj_mad",
                "instance_obj_std",
                "instance_log_prob_std",
                "instance_regret_std",
            ],
            returns="PrefBatch",
            mode="pairwise",
        ),
        hyperparams=hyper_out,
        operators_used=["all_pairs_reweight", str(weight_family)],
        code="\n".join(code_lines) + "\n",
    )


def _loss_ir_from_template(obj: Mapping[str, Any]) -> FreeLossIR:
    hyper = _mapping_hyperparams(obj)
    template = str(obj.get("template") or hyper.get("template") or "weighted_softplus_gap").strip().lower()
    if template not in _LOSS_TEMPLATES:
        template = "weighted_softplus_gap"
    temperature = _safe_float(hyper.get("temperature", 1.0), 1.0, lo=0.05, hi=8.0)
    gap_scale = _safe_float(hyper.get("gap_scale", 0.5), 0.5, lo=0.0, hi=8.0)
    target_margin = _safe_float(hyper.get("target_margin", 0.0), 0.0, lo=0.0, hi=8.0)

    if template == "weighted_softplus_rank":
        signal_line = "    signal = batch.get('delta_rank', torch.zeros_like(margin)).detach().clamp_min(0.0)"
        loss_line = f"    loss_vec = weight * F.softplus(({target_margin:.8g} + {gap_scale:.8g} * signal) - {temperature:.8g} * margin)"
        expects = ["log_prob_w", "log_prob_l", "delta_rank", "weight"]
    elif template == "weighted_logsigmoid_gap":
        signal_line = "    signal = batch.get('delta_z', torch.zeros_like(margin)).detach().clamp_min(0.0)"
        loss_line = f"    loss_vec = -weight * F.logsigmoid({temperature:.8g} * (margin - ({target_margin:.8g} + {gap_scale:.8g} * signal)))"
        expects = ["log_prob_w", "log_prob_l", "delta_z", "weight"]
    else:
        signal_line = "    signal = batch.get('delta_z', torch.zeros_like(margin)).detach().clamp_min(0.0)"
        loss_line = f"    loss_vec = weight * F.softplus(({target_margin:.8g} + {gap_scale:.8g} * signal) - {temperature:.8g} * margin)"
        expects = ["log_prob_w", "log_prob_l", "delta_z", "weight"]

    code = "\n".join(
        [
            "def generated_loss(batch, model_output, extra):",
            "    log_prob_w = batch['log_prob_w']",
            "    log_prob_l = batch['log_prob_l']",
            "    margin = log_prob_w - log_prob_l",
            "    weight = batch.get('weight', torch.ones_like(margin))",
            "    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)",
            signal_line,
            loss_line,
            "    denom = weight.sum().clamp_min(1.0)",
            "    return loss_vec.sum() / denom",
        ]
    ) + "\n"
    name = _safe_name(obj.get("name"), f"loss_template_{template}")
    hyper_out = dict(hyper)
    hyper_out.update(
        {
            "template": str(template),
            "temperature": float(temperature),
            "gap_scale": float(gap_scale),
            "target_margin": float(target_margin),
            "paradigm_family": "pairwise_margin",
            "agg_family": "weighted_mean",
            "constraint_family": "finite_softplus",
        }
    )
    return FreeLossIR(
        name=name,
        intuition=str(obj.get("intuition") or f"template pairwise weighted loss using {template}"),
        pseudocode=str(obj.get("pseudocode") or "Compute a weighted pairwise margin loss over builder-selected pairs."),
        hyperparams=hyper_out,
        operators_used=[str(template), "weighted_mean"],
        implementation_hint=FreeLossImplementationHint(
            expects=expects,
            returns="scalar",
            mode="pairwise",
        ),
        code=code,
        theoretical_basis=str(obj.get("theoretical_basis") or ""),
    )


def parse_joint_pair_from_text(
    text: str,
    *,
    template_constraints: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, FreeLossIR]:
    json_str = _extract_json_object(text)
    obj = json.loads(json_str)
    if not isinstance(obj, Mapping):
        raise ValueError("Joint pair JSON must be an object")

    constraints = _normalize_template_constraints(template_constraints)
    builder_obj = dict(_extract_mapping_field(obj, ("builder", "preference_builder", "pref_builder", "g")))
    loss_obj = dict(_extract_mapping_field(obj, ("loss", "free_loss", "preference_loss", "f")))
    builder_obj = _with_template_constraints(
        builder_obj,
        target_weight_family=constraints.get("target_weight_family"),
    )
    loss_obj = _with_template_constraints(
        loss_obj,
        target_loss_template=constraints.get("target_loss_template"),
    )

    builder_code = str(builder_obj.get("code", "") or "")
    if (not bool(constraints.get("force_template", True))) and "def generated_builder" in builder_code:
        builder_ir = _parse_pref_builder_from_text(json.dumps(dict(builder_obj), ensure_ascii=False))
    else:
        builder_ir = _builder_ir_from_template(builder_obj)

    loss_code = str(loss_obj.get("code", "") or "")
    if (not bool(constraints.get("force_template", True))) and "def generated_loss" in loss_code:
        loss_ir = parse_free_loss_from_text(json.dumps(dict(loss_obj), ensure_ascii=False))
    else:
        loss_ir = _loss_ir_from_template(loss_obj)
    return builder_ir, loss_ir


def build_joint_pair_prompt(
    *,
    prompt_path: str,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    if isinstance(global_feedback, Mapping):
        prompt = _append_global_feedback(prompt, global_feedback)
    prompt_hash = sha1(prompt.encode("utf-8")).hexdigest()
    return prompt, prompt_hash


def generate_joint_pair_candidate_with_meta(
    *,
    prompt_path: str,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
    template_constraints: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, FreeLossIR, Mapping[str, Any]]:
    prompt, prompt_hash = build_joint_pair_prompt(
        prompt_path=prompt_path,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="JOINT_PAIR_GENERATE", prompt_path=prompt_path)
    builder_ir, loss_ir = parse_joint_pair_from_text(raw, template_constraints=template_constraints)
    return (
        builder_ir,
        loss_ir,
        {
            "llm_op": "JOINT_PAIR_GENERATE",
            "prompt_path": prompt_path,
            "prompt_sha1": prompt_hash,
        },
    )
