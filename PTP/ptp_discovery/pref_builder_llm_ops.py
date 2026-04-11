from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .free_loss_llm_ops import _call_llm, _extract_json_object, configure_llm_run
from .pref_builder_ir import PreferenceBuilderIR, ir_from_json as pref_builder_ir_from_json


LOGGER = logging.getLogger(__name__)

_BUILDER_OPTIONAL_FEATURE_KEYS = {
    "seq_len",
    "log_prob_mean",
    "advantage",
    "entropy",
    "entropy_mean",
    "log_prob_step",
}


def _sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def _read_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        name = os.path.basename(path)
        LOGGER.warning("Prompt file missing (%s); using built-in fallback prompt.", path)
        return _fallback_prompt(name)


def _fallback_prompt(name: str) -> str:
    base = """You are generating a single JSON object for a preference builder candidate.

Return ONLY a JSON object. It must match this schema:
{
  "name": "...",
  "intuition": "...",
  "hyperparams": {},
  "operators_used": ["..."],
  "implementation_hint": {
    "expects": ["objective", "log_prob"],
    "returns": "PrefBatch",
    "mode": "pairwise"
  },
  "code": "def generated_builder(feature_cache, extra):\\n    ...\\n"
}

Constraints:
- generated_builder must return a PrefBatch.
- Do not use imports; do not access filesystem; no eval/exec/open.
- You may use torch, F, ops, and PrefBatch which are provided by the sandbox.
- Use vectorized tensor ops only; do not use Python loops/comprehensions over pair indices.
- Do not build O(num_pairs) Python loops for capping/filtering; prefer tensor masking/topk/slicing.
"""
    n = str(name or "").strip().lower()
    if "repair" in n:
        return base + "\nTask: Repair the candidate to satisfy gates and keep the same output schema."
    if "crossover" in n:
        return base + "\nTask: Combine the best ideas from the provided parents to produce a new child candidate."
    if "mutation" in n or n in {"m2", "m3"}:
        return base + "\nTask: Mutate the provided parent candidate to produce a new child candidate."
    if "e2" in n:
        return base + "\nTask: Produce a novel candidate distinct from the parents while staying within the schema."
    return base


def _parse_pref_builder_from_text(text: str) -> PreferenceBuilderIR:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Builder JSON must be an object")

    code = str(obj.get("code", "") or "").strip()
    if not code:
        raise ValueError("Builder JSON missing code")
    if "def generated_builder" not in code:
        raise ValueError("Builder code must define generated_builder")

    impl = obj.get("implementation_hint", {}) or {}
    if not isinstance(impl, dict):
        impl = {}
    expects = impl.get("expects", ["objective", "log_prob"]) or ["objective", "log_prob"]
    if not isinstance(expects, (list, tuple)):
        expects = [str(expects)]
    expects = [str(x) for x in expects if str(x).strip()]
    if not expects:
        expects = ["objective", "log_prob"]

    mode = str(impl.get("mode", "pairwise") or "pairwise").strip().lower()
    if mode not in {"pairwise", "setwise", "listwise"}:
        mode = "pairwise"
    impl["expects"] = expects
    impl["returns"] = str(impl.get("returns", "PrefBatch") or "PrefBatch")
    impl["mode"] = mode
    obj["implementation_hint"] = impl

    name = str(obj.get("name", "") or "").strip() or "unnamed_preference_builder"
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)[:80]
    obj["name"] = name

    return pref_builder_ir_from_json(obj)


def build_runtime_prompt_context(
    *,
    loss_observables: Sequence[str] | None,
    mode: str = "pairwise",
) -> Mapping[str, Any]:
    mode_norm = str(mode or "pairwise").strip().lower() or "pairwise"
    observable_names = [str(v).strip() for v in (loss_observables or []) if str(v).strip()]
    observable_set = set(observable_names)

    base_keys = ["objective", "log_prob", "obj_z", "rank", "regret"]
    available = set(base_keys)
    preferred = set(base_keys)
    blocked = set()

    for key in sorted(_BUILDER_OPTIONAL_FEATURE_KEYS):
        if key in observable_set:
            available.add(key)
        else:
            blocked.add(key)

    return {
        "mode": mode_norm,
        "configured_loss_observables": observable_names,
        "available_keys": sorted(available),
        "preferred_cheap_keys": sorted(preferred),
        "blocked_optional_keys": sorted(blocked),
        "notes": [
            "implementation_hint.expects and required feature_cache[...] access must stay within available_keys",
            "optional signals should be accessed with feature_cache.get(..., fallback)",
            "prefer objective/log_prob/obj_z/rank/regret unless an optional observable is explicitly available",
        ],
    }


def _append_prompt_context_block(prompt: str, prompt_context: Mapping[str, Any] | None) -> str:
    if not prompt_context:
        return prompt
    return (
        prompt
        + "\n\nRUNTIME_CONTEXT_JSON:\n"
        + json.dumps(dict(prompt_context), indent=2, ensure_ascii=False)
        + "\n\nFollow RUNTIME_CONTEXT_JSON strictly. Do not require keys listed in "
        + "`blocked_optional_keys`."
    )


def _append_global_feedback(prompt: str, global_feedback: Mapping[str, Any] | None) -> str:
    if global_feedback is None:
        return prompt
    out = prompt
    llm_call = global_feedback.get("llm_call") if isinstance(global_feedback, Mapping) else None
    search_space = global_feedback.get("builder_search_space") if isinstance(global_feedback, Mapping) else None
    if isinstance(search_space, Mapping):
        mode = str(search_space.get("mode", "") or "").strip().lower()
        if mode == "reweight_only":
            fixed_pair_builder = str(search_space.get("fixed_pair_builder", "all_pairs") or "all_pairs").strip().lower()
            pair_weight_normalization = str(
                search_space.get("pair_weight_normalization", "instance_mean") or "instance_mean"
            ).strip().lower()
            allowed_weight_families = search_space.get("allowed_weight_families", [])
            if not isinstance(allowed_weight_families, (list, tuple)):
                allowed_weight_families = []
            seed_weight_families = search_space.get("seed_weight_families", [])
            if not isinstance(seed_weight_families, (list, tuple)):
                seed_weight_families = []
            allow_freeform_weight_family = bool(search_space.get("allow_freeform_weight_family", False))
            out += (
                "\n\nBUILDER_SEARCH_SPACE_CONSTRAINTS:\n"
                "- Search mode is reweight_only.\n"
                f"- You must preserve the pair construction of the fixed template `{fixed_pair_builder}`.\n"
                "- Fixed all-pair topology preserves the maximum amount of pair information; do not throw that away.\n"
                "- Do not change pair topology, candidate selection, coverage pattern, or pair capping logic.\n"
                "- Your only substantive degree of freedom is the nonnegative pair weight function.\n"
                "- Keep pair_idx identical to the fixed template and modify only `weight` plus metadata/hyperparameters.\n"
                "- Weight must be finite, nonnegative, vectorized, and instance-local.\n"
                "- This is not redundant with loss-only search: the downstream loss batch is flattened and does not carry `b_idx`, so it cannot reconstruct instance-local pair distributions or per-instance pool statistics.\n"
                "- Prefer configurable scalars via `extra` such as weight_tau or weight_beta.\n"
            )
            if pair_weight_normalization == "none":
                out += (
                    "- Pair-weight normalization mode is `none`.\n"
                    "- Do not divide weights by per-instance sums or means.\n"
                    "- Use raw nonnegative weighting followed by explicit clamping only.\n"
                    "- Let the downstream loss-side weighted mean handle global scale normalization.\n"
                )
            else:
                out += (
                    "- Pair-weight normalization mode is `instance_mean`.\n"
                    "- Use weighting to reshape the per-instance pair distribution with full-pool context and instance-local normalization.\n"
                )
            if allow_freeform_weight_family:
                out += (
                    f"- Seed weight_family values for the handcrafted initial pool: {json.dumps([str(x) for x in seed_weight_families], ensure_ascii=False)}\n"
                    "- In this mode, `weight_family` is descriptive metadata rather than a hard whitelist.\n"
                    "- You may introduce a new `weight_family` label if the actual formula is genuinely new.\n"
                )
            else:
                out += f"- Allowed weight_family values: {json.dumps([str(x) for x in allowed_weight_families], ensure_ascii=False)}\n"
            op_name = ""
            if isinstance(llm_call, Mapping):
                op_name = str(llm_call.get("search_operator") or llm_call.get("op_type") or "").strip().upper()
            if op_name:
                out += "\nREWEIGHT_ONLY_OPERATOR_SEMANTICS:\n"
                if op_name == "PARADIGM_SHIFT":
                    out += (
                        "- This is a cross-family transfer step for weighting search.\n"
                        "- Preserve the fixed pair template exactly.\n"
                        "- Change `weight_family` relative to the dominant parent family so underrepresented paradigms can continue evolving.\n"
                        "- Make a real weighting-form change, not just a scalar retune.\n"
                    )
                elif op_name == "STRUCTURE_SHIFT":
                    out += (
                        "- This is a signal-organization rewrite under the same successful weighting family.\n"
                        "- Preserve `weight_family` and `constraint_family` unless correctness forces otherwise.\n"
                        "- Rewrite clipping, rescaling, ranking, or gap-to-weight transformation structure.\n"
                        "- Respect the configured pair-weight normalization mode.\n"
                        "- Do not reduce this to a scalar-only tune.\n"
                    )
                elif op_name == "CONSTRAINT_INJECT":
                    out += (
                        "- This step should add explicit optimization/stability constraints while preserving the core weighting family.\n"
                        "- Preserve `weight_family` and fixed pair topology.\n"
                        "- Add concrete stabilizers such as denominator safeguards, clamp, topk caps, or tie-zone filtering.\n"
                        "- Do not add instance-local normalization when the configured pair-weight normalization mode is `none`.\n"
                    )
                elif op_name in {"XOVER", "E1"}:
                    out += (
                        "- This is crossover over existing weighting candidates.\n"
                        "- Preserve the fixed pair template and combine complementary weighting ideas from multiple parents.\n"
                        "- Reuse effective substructures already validated in the parents when possible.\n"
                    )
                elif op_name in {"TUNE", "M2"}:
                    out += (
                        "- This is local exploitation.\n"
                        "- Keep the overall weighting logic the same and tune only local hyperparameters or smooth scalar transforms.\n"
                        "- Avoid introducing a new weighting family unless it is absolutely necessary for correctness.\n"
                    )
    return out + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)


def build_generation_prompt(
    generation_prompt_path: str,
    *,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(generation_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_crossover_prompt(
    crossover_prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(crossover_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "implementation_hint": asdict(parent.implementation_hint),
                "code": parent.code,
                "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
            }
        )
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_e2_prompt(
    e2_prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(e2_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "implementation_hint": asdict(parent.implementation_hint),
                "code": parent.code,
                "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
            }
        )
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_mutation_prompt(
    mutation_prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(mutation_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "implementation_hint": asdict(parent.implementation_hint),
        "code": parent.code,
        "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_m2_prompt(
    m2_prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(m2_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "implementation_hint": asdict(parent.implementation_hint),
        "code": parent.code,
        "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_paradigm_shift_prompt(
    prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "implementation_hint": asdict(parent.implementation_hint),
                "code": parent.code,
                "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
            }
        )
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(blobs, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_structure_shift_prompt(
    prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "implementation_hint": asdict(parent.implementation_hint),
        "code": parent.code,
        "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_constraint_inject_prompt(
    prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    metrics: Mapping[str, Any] = parent_fitness or {}
    blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "implementation_hint": asdict(parent.implementation_hint),
        "code": parent.code,
        "metrics": {"fitness": float(metrics.get("fitness", float("inf"))) if metrics else None},
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(blob, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_m3_prompt(
    m3_prompt_path: str,
    *,
    candidate: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(m3_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    payload = {
        "candidate": {
            "name": candidate.name,
            "intuition": candidate.intuition,
            "hyperparams": candidate.hyperparams,
            "operators_used": candidate.operators_used,
            "implementation_hint": asdict(candidate.implementation_hint),
            "code": candidate.code,
        },
        "failure_reason": dict(failure_reason),
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_repair_prompt(
    repair_prompt_path: str,
    *,
    failed_ir: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(repair_prompt_path)
    prompt = _append_prompt_context_block(prompt, prompt_context)
    payload = {
        "candidate": {
            "name": failed_ir.name,
            "intuition": failed_ir.intuition,
            "hyperparams": failed_ir.hyperparams,
            "operators_used": failed_ir.operators_used,
            "implementation_hint": asdict(failed_ir.implementation_hint),
            "code": failed_ir.code,
        },
        "failure_reason": dict(failure_reason),
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def generate_pref_builder_candidate(
    generation_prompt_path: str,
    *,
    operator_whitelist: Sequence[str],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    del operator_whitelist
    prompt, _ = build_generation_prompt(
        generation_prompt_path,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E1_GENERATE", prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def crossover_pref_builder(
    crossover_prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_crossover_prompt(
        crossover_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E1", prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def mutate_pref_builder(
    mutation_prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_mutation_prompt(
        mutation_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M1", prompt_path=mutation_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def e2_pref_builder(
    e2_prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_e2_prompt(
        e2_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E2", prompt_path=e2_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def m2_tune_builder(
    m2_prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_m2_prompt(
        m2_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M2", prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def paradigm_shift_builder(
    prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_paradigm_shift_prompt(
        prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_PARADIGM_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def structure_shift_builder(
    prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_structure_shift_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_STRUCTURE_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def constraint_inject_builder(
    prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_constraint_inject_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_CONSTRAINT_INJECT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def m3_simplify_builder(
    m3_prompt_path: str,
    candidate: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_m3_prompt(
        m3_prompt_path,
        candidate=candidate,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M3", prompt_path=m3_prompt_path)
    json_str = _extract_json_object(raw)
    out = _parse_pref_builder_from_text(json_str)
    out.implementation_hint = candidate.implementation_hint
    return out


def repair_pref_builder(
    repair_prompt_path: str,
    failed_ir: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_repair_prompt(
        repair_prompt_path,
        failed_ir=failed_ir,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="REPAIR", prompt_path=repair_prompt_path)
    json_str = _extract_json_object(raw)
    out = _parse_pref_builder_from_text(json_str)
    out.implementation_hint = failed_ir.implementation_hint
    return out


def generate_pref_builder_candidate_with_meta(
    generation_prompt_path: str,
    *,
    operator_whitelist: Sequence[str],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_generation_prompt(
        generation_prompt_path,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E1_GENERATE", prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "E1_GENERATE",
            "prompt_path": str(generation_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def crossover_pref_builder_with_meta(
    crossover_prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_crossover_prompt(
        crossover_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E1", prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "E1",
            "prompt_path": str(crossover_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def e2_pref_builder_with_meta(
    e2_prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_e2_prompt(
        e2_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="E2", prompt_path=e2_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "E2",
            "prompt_path": str(e2_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def mutate_pref_builder_with_meta(
    mutation_prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_mutation_prompt(
        mutation_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M1", prompt_path=mutation_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "M1",
            "prompt_path": str(mutation_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def m2_tune_builder_with_meta(
    m2_prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_m2_prompt(
        m2_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M2", prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "M2",
            "prompt_path": str(m2_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def paradigm_shift_builder_with_meta(
    prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_paradigm_shift_prompt(
        prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_PARADIGM_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "BUILDER_PARADIGM_SHIFT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def structure_shift_builder_with_meta(
    prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_structure_shift_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_STRUCTURE_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "BUILDER_STRUCTURE_SHIFT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def constraint_inject_builder_with_meta(
    prompt_path: str,
    *,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_constraint_inject_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_CONSTRAINT_INJECT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    ir = _parse_pref_builder_from_text(json_str)
    return (
        ir,
        {
            "llm_op": "BUILDER_CONSTRAINT_INJECT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def m3_simplify_builder_with_meta(
    m3_prompt_path: str,
    *,
    candidate: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_m3_prompt(
        m3_prompt_path,
        candidate=candidate,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="M3", prompt_path=m3_prompt_path)
    json_str = _extract_json_object(raw)
    out = _parse_pref_builder_from_text(json_str)
    out.implementation_hint = candidate.implementation_hint
    return (
        out,
        {
            "llm_op": "M3",
            "prompt_path": str(m3_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def repair_pref_builder_with_meta(
    repair_prompt_path: str,
    *,
    failed_ir: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
    prompt_context: Mapping[str, Any] | None = None,
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_repair_prompt(
        repair_prompt_path,
        failed_ir=failed_ir,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
        prompt_context=prompt_context,
    )
    raw = _call_llm(prompt, llm_op="REPAIR", prompt_path=repair_prompt_path)
    json_str = _extract_json_object(raw)
    out = _parse_pref_builder_from_text(json_str)
    out.implementation_hint = failed_ir.implementation_hint
    return (
        out,
        {
            "llm_op": "REPAIR",
            "prompt_path": str(repair_prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )
