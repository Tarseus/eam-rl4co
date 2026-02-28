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


def _append_global_feedback(prompt: str, global_feedback: Mapping[str, Any] | None) -> str:
    if global_feedback is None:
        return prompt
    return prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)


def build_generation_prompt(
    generation_prompt_path: str,
    *,
    global_feedback: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(generation_prompt_path)
    prompt = _append_global_feedback(prompt, global_feedback)
    return prompt, _sha1(prompt)


def build_crossover_prompt(
    crossover_prompt_path: str,
    *,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    prompt = _read_prompt(crossover_prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(e2_prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(mutation_prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(m2_prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(m3_prompt_path)
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
) -> tuple[str, str]:
    prompt = _read_prompt(repair_prompt_path)
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
) -> PreferenceBuilderIR:
    del operator_whitelist
    prompt, _ = build_generation_prompt(generation_prompt_path, global_feedback=global_feedback)
    raw = _call_llm(prompt, llm_op="E1_GENERATE", prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def crossover_pref_builder(
    crossover_prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_crossover_prompt(
        crossover_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="E1", prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def mutate_pref_builder(
    mutation_prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_mutation_prompt(
        mutation_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="M1", prompt_path=mutation_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def e2_pref_builder(
    e2_prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_e2_prompt(
        e2_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="E2", prompt_path=e2_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def m2_tune_builder(
    m2_prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_m2_prompt(
        m2_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="M2", prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def paradigm_shift_builder(
    prompt_path: str,
    parents: Sequence[PreferenceBuilderIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_paradigm_shift_prompt(
        prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_PARADIGM_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def structure_shift_builder(
    prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_structure_shift_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_STRUCTURE_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def constraint_inject_builder(
    prompt_path: str,
    parent: PreferenceBuilderIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_constraint_inject_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
    )
    raw = _call_llm(prompt, llm_op="BUILDER_CONSTRAINT_INJECT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return _parse_pref_builder_from_text(json_str)


def m3_simplify_builder(
    m3_prompt_path: str,
    candidate: PreferenceBuilderIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
) -> PreferenceBuilderIR:
    prompt, _ = build_m3_prompt(
        m3_prompt_path,
        candidate=candidate,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
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
) -> PreferenceBuilderIR:
    prompt, _ = build_repair_prompt(
        repair_prompt_path,
        failed_ir=failed_ir,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_generation_prompt(generation_prompt_path, global_feedback=global_feedback)
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_crossover_prompt(
        crossover_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_e2_prompt(
        e2_prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_mutation_prompt(
        mutation_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_m2_prompt(
        m2_prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_paradigm_shift_prompt(
        prompt_path,
        parents=parents,
        parents_fitness=parents_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_structure_shift_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_constraint_inject_prompt(
        prompt_path,
        parent=parent,
        parent_fitness=parent_fitness,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_m3_prompt(
        m3_prompt_path,
        candidate=candidate,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
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
) -> tuple[PreferenceBuilderIR, Mapping[str, Any]]:
    prompt, prompt_sha1 = build_repair_prompt(
        repair_prompt_path,
        failed_ir=failed_ir,
        failure_reason=failure_reason,
        global_feedback=global_feedback,
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
