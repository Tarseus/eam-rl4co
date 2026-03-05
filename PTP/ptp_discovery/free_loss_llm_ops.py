from __future__ import annotations

import json
import logging
import os
import random
import re
import shlex
import time
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha1
from typing import Any, Mapping, Sequence, TYPE_CHECKING

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # noqa: BLE001
    load_dotenv = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from openai import OpenAI  # noqa: F401

from .free_loss_compiler import (
    CompiledFreeLoss,
    compile_free_loss,
    parse_free_loss_from_text,
)
from .free_loss_ir import FreeLossIR


LOGGER = logging.getLogger(__name__)
_OPENAI_CLIENT: Any | None = None
_ENV_LOADED = False
_OFFLINE_MODE = False

# Run-level LLM cache (prompt+inputs hash) to reduce repeated calls on resume/re-runs.
_LLM_CACHE_PATH: str | None = None
_LLM_CACHE_INDEX: dict[str, str] = {}
_LLM_CACHE_HITS = 0
_LLM_CACHE_MISSES = 0


def _repo_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../PTP/ptp_discovery
    return os.path.abspath(os.path.join(this_dir, "..", ".."))  # .../<repo_root>


def _dotenv_candidates() -> list[str]:
    candidates: list[str] = []
    explicit = str(os.getenv("OPENAI_DOTENV_PATH", "") or "").strip()
    if explicit:
        candidates.append(os.path.abspath(explicit))
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), ".env")))
    candidates.append(os.path.abspath(os.path.join(_repo_root(), ".env")))

    unique: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_dotenv_fallback(path: str) -> int:
    """Minimal .env loader used when python-dotenv is unavailable."""

    loaded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or any(ch.isspace() for ch in key):
                    continue
                if value and value[0] in {"'", '"'}:
                    try:
                        parsed = shlex.split(value, posix=True)
                    except ValueError:
                        parsed = []
                    if parsed:
                        value = parsed[0]
                else:
                    value = value.split(" #", 1)[0].strip()
                if key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except OSError:
        return loaded
    return loaded


def configure_llm_run(*, run_dir: str | None = None, cache_path: str | None = None, offline_mode: bool | None = None) -> None:
    """Configure run-scoped LLM settings (cache path + offline mode).

    Intended to be called once from the EoH loop after `run_dir` is known.
    """

    global _LLM_CACHE_PATH, _LLM_CACHE_INDEX, _OFFLINE_MODE, _LLM_CACHE_HITS, _LLM_CACHE_MISSES

    if offline_mode is not None:
        _OFFLINE_MODE = bool(offline_mode)

    if cache_path is None and run_dir:
        cache_path = os.path.join(str(run_dir), "llm_cache.jsonl")
    if cache_path:
        _LLM_CACHE_PATH = str(cache_path)
        _LLM_CACHE_INDEX = {}
        _LLM_CACHE_HITS = 0
        _LLM_CACHE_MISSES = 0
        _load_llm_cache()


def llm_cache_stats() -> Mapping[str, Any]:
    return {
        "offline_mode": bool(_OFFLINE_MODE),
        "cache_path": _LLM_CACHE_PATH,
        "cache_entries": int(len(_LLM_CACHE_INDEX)),
        "cache_hits": int(_LLM_CACHE_HITS),
        "cache_misses": int(_LLM_CACHE_MISSES),
    }


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if _OFFLINE_MODE:
        _ENV_LOADED = True
        return
    dotenv_candidates = _dotenv_candidates()
    loaded_dotenv_path: str | None = None
    if load_dotenv is not None:
        for path in dotenv_candidates:
            if os.path.isfile(path):
                load_dotenv(dotenv_path=path, override=False)
                loaded_dotenv_path = path
                break
    else:
        for path in dotenv_candidates:
            if os.path.isfile(path):
                loaded = _load_dotenv_fallback(path)
                loaded_dotenv_path = path
                LOGGER.warning(
                    "python-dotenv is unavailable; loaded %d key(s) via fallback parser from %s",
                    loaded,
                    path,
                )
                break
    if not os.getenv("OPENAI_API_KEY"):
        searched = ", ".join(dotenv_candidates)
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment (or a .env at repo root), "
            f"searched .env paths: [{searched}], "
            "or run in offline_mode=true. If you don't have the EoH deps installed, run: "
            "pip install -e '.[eoh]'."
        )
    if loaded_dotenv_path:
        LOGGER.info("Loaded LLM environment from .env: %s", loaded_dotenv_path)
    _ENV_LOADED = True


@lru_cache(maxsize=1)
def _openai_symbols() -> tuple[Any, tuple[type[BaseException], ...], type[BaseException] | None]:
    """Import OpenAI SDK symbols lazily so normal RL4CO usage doesn't require them."""

    try:
        from openai import (  # type: ignore
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            OpenAI,
            RateLimitError,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "openai package is not installed. Install EoH extras with: pip install -e '.[eoh]'."
        ) from exc
    retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    return OpenAI, retryable, BadRequestError


def _make_openai_client() -> Any:
    timeout_s = float(os.getenv("OPENAI_TIMEOUT_S", "60") or 60)
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2") or 2)
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OpenAI, _, _ = _openai_symbols()
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries)


def _get_openai_client() -> Any:
    global _OPENAI_CLIENT
    _load_env()
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = _make_openai_client()
    return _OPENAI_CLIENT


def _should_retry_llm_error(exc: Exception) -> bool:
    # Treat transient transport/service issues as retryable. Some providers/proxies
    # return "get_token_error" as a 500; this is usually transient as well.
    try:
        _, retryable_types, bad_request = _openai_symbols()
    except Exception:  # noqa: BLE001
        return False
    if isinstance(exc, retryable_types):
        return True
    if bad_request is not None and isinstance(exc, bad_request) and "get_token_error" in str(exc):
        return True
    return False


def _read_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # The repo historically referenced `PTP/prompts/*.txt` from configs, but
        # some distributions do not ship the prompt assets. To avoid a hard
        # crash, fall back to a minimal built-in prompt that preserves the JSON
        # contract required by the pipeline.
        name = os.path.basename(path)
        LOGGER.warning("Prompt file missing (%s); using built-in fallback prompt.", path)
        return _fallback_prompt(name)


def _fallback_prompt(name: str) -> str:
    base = """You are generating a single JSON object for a free-form preference loss candidate.

Return ONLY a JSON object. It must match this schema:
{
  "name": "...",
  "intuition": "...",
  "pseudocode": "...",
  "hyperparams": {},
  "operators_used": ["..."],
  "implementation_hint": {
    "expects": ["log_prob_w", "log_prob_l", "delta_z", "weight"],
    "returns": "scalar",
    "mode": "pairwise"
  },
  "code": "def generated_loss(batch, model_output, extra):\\n    ...\\n"
}

Constraints:
- `generated_loss` must return a scalar torch.Tensor.
- Do not use imports; do not access filesystem; no eval/exec/open.
- Use vectorized tensor ops only; do not use Python loops/comprehensions over batch/pair tensors.
"""

    n = str(name or "").strip().lower()
    if "expects" in n:
        return (
            base
            + "\nTask: Repair/normalize implementation_hint.expects so it is a non-empty list consistent with mode."
        )
    if "repair" in n:
        return base + "\nTask: Repair the candidate to satisfy gates and keep the same output schema."
    if "crossover" in n:
        return base + "\nTask: Combine the best ideas from the provided parents to produce a new child candidate."
    if "mutation" in n or n in {"m2", "m3"}:
        return base + "\nTask: Mutate the provided parent candidate to produce a new child candidate."
    if "e2" in n:
        return base + "\nTask: Produce a novel candidate distinct from the parents while staying within the schema."
    return base


def _extract_json_object(text: str) -> str:
    """Extract the first complete top-level JSON object from model output.

    More robust than slicing from first '{' to last '}' because LLMs may emit
    multiple objects or trailing text.
    """

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    end = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None or end <= start:
        raise ValueError("Failed to locate a complete JSON object in model output.")

    snippet = text[start : end + 1]

    invalid_escape_pattern = re.compile(r'\\(?!["\\/bfnrtu])')
    sanitized = invalid_escape_pattern.sub(r"\\\\", snippet)

    def _escape_control_chars_in_strings(s: str) -> str:
        out_chars: list[str] = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                out_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                out_chars.append(ch)
                escape = True
                continue
            if ch == '"':
                out_chars.append(ch)
                in_string = not in_string
                continue
            if in_string and ch in ("\n", "\r", "\t"):
                if ch == "\n":
                    out_chars.append("\\n")
                elif ch == "\r":
                    out_chars.append("\\r")
                else:
                    out_chars.append("\\t")
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    return _escape_control_chars_in_strings(sanitized)


def _load_llm_cache() -> None:
    global _LLM_CACHE_INDEX
    if not _LLM_CACHE_PATH:
        return
    path = str(_LLM_CACHE_PATH)
    if not os.path.isfile(path):
        return
    loaded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(rec, dict):
                    continue
                key = rec.get("key")
                content = rec.get("content")
                if isinstance(key, str) and isinstance(content, str):
                    _LLM_CACHE_INDEX[key] = content
                    loaded += 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load LLM cache (%s): %s", path, exc)
        return
    if loaded:
        LOGGER.info("Loaded LLM cache entries: %d (%s)", loaded, path)


def _append_llm_cache_record(record: Mapping[str, Any]) -> None:
    if not _LLM_CACHE_PATH:
        return
    path = str(_LLM_CACHE_PATH)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to append LLM cache record (%s): %s", path, exc)


def _cache_key(*, model: str, prompt: str) -> str:
    blob = json.dumps({"model": str(model), "prompt": str(prompt)}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha1(blob).hexdigest()


def _resolve_llm_model_for_op(llm_op: str) -> str:
    op = str(llm_op or "").strip().upper()

    nano_default = os.getenv("OPENAI_MODEL_NANO", "gpt-4.1-nano")
    mini_default = os.getenv("OPENAI_MODEL_MINI", "gpt-4.1-mini")
    fallback_model = os.getenv("OPENAI_MODEL", mini_default)

    nano_ops = {"E1_GENERATE", "E1", "E2", "M1", "M2"}
    mini_prefixes = ("DIR_REPAIR_",)
    mini_suffixes = ("_PARADIGM_SHIFT", "_STRUCTURE_SHIFT", "_CONSTRAINT_INJECT")
    mini_ops = {"REPAIR", "EXPECTS_REPAIR", "M3"}

    if op in nano_ops:
        return str(nano_default)
    if op in mini_ops:
        return str(mini_default)
    if any(op.startswith(prefix) for prefix in mini_prefixes):
        return str(mini_default)
    if any(op.endswith(suffix) for suffix in mini_suffixes):
        return str(mini_default)
    return str(fallback_model)


def _call_llm(prompt: str, *, llm_op: str, prompt_path: str | None) -> str:
    global _LLM_CACHE_HITS, _LLM_CACHE_MISSES

    if _OFFLINE_MODE:
        raise RuntimeError("offline_mode=true: LLM calls are disabled for this run.")

    client = _get_openai_client()
    model_name = _resolve_llm_model_for_op(llm_op)

    key = _cache_key(model=model_name, prompt=prompt)
    if _LLM_CACHE_PATH and key in _LLM_CACHE_INDEX:
        _LLM_CACHE_HITS += 1
        return _LLM_CACHE_INDEX[key]

    _LLM_CACHE_MISSES += 1

    max_attempts = int(os.getenv("OPENAI_CALL_MAX_ATTEMPTS", "6") or 6)
    base_backoff_s = float(os.getenv("OPENAI_CALL_BACKOFF_S", "1") or 1)
    max_backoff_s = float(os.getenv("OPENAI_CALL_BACKOFF_MAX_S", "30") or 30)

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = resp.choices[0].message.content
            if not content:
                raise RuntimeError("LLM returned empty content.")
            out = content.strip()
            if _LLM_CACHE_PATH:
                _LLM_CACHE_INDEX[key] = out
                _append_llm_cache_record(
                    {
                        "key": key,
                        "ts": float(time.time()),
                        "model": str(model_name),
                        "llm_op": str(llm_op),
                        "prompt_path": str(prompt_path) if prompt_path else None,
                        "prompt_sha1": sha1(prompt.encode("utf-8")).hexdigest(),
                        "content": out,
                    }
                )
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts or not _should_retry_llm_error(exc):
                raise

            sleep_s = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
            sleep_s = sleep_s * (0.5 + random.random())  # jitter
            LOGGER.warning(
                "LLM call failed (attempt %d/%d, model=%s): %s; retrying in %.1fs",
                attempt,
                max_attempts,
                model_name,
                str(exc),
                sleep_s,
            )
            time.sleep(sleep_s)

    # Should be unreachable, but keeps types happy.
    raise RuntimeError("LLM call failed after retries.") from last_exc


def generate_free_loss_candidate(
    generation_prompt_path: str,
    *,
    operator_whitelist: Sequence[str],
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    del operator_whitelist
    base_prompt = _read_prompt(generation_prompt_path)
    prompt = base_prompt
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt, llm_op="E1_GENERATE", prompt_path=generation_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def crossover_free_loss(
    crossover_prompt_path: str,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(crossover_prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blob = {
            "index": idx,
            "name": parent.name,
            "intuition": parent.intuition,
            "pseudocode": parent.pseudocode,
            "hyperparams": parent.hyperparams,
            "operators_used": parent.operators_used,
            "code": parent.code,
            "theoretical_basis": getattr(parent, "theoretical_basis", ""),
            "metrics": {
                "hf_like_score": float(metrics.get("hf_like_score", float("inf")))
                if metrics
                else None,
                "validation_objective": float(metrics.get("validation_objective", float("inf")))
                if metrics
                else None,
                "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
                if metrics
                else None,
                "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            },
        }
        parent_blobs.append(blob)
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt, llm_op="E1", prompt_path=crossover_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def mutate_free_loss(
    mutation_prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(mutation_prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
            if metrics
            else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob
    raw = _call_llm(prompt, llm_op="M1", prompt_path=mutation_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def e2_free_loss(
    e2_prompt_path: str,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """E2: consensus extraction over p parents, then synthesize a new child loss."""

    prompt = _read_prompt(e2_prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        blob = {
            "index": idx,
            "name": parent.name,
            "intuition": parent.intuition,
            "pseudocode": parent.pseudocode,
            "hyperparams": parent.hyperparams,
            "operators_used": parent.operators_used,
            "code": parent.code,
            "theoretical_basis": getattr(parent, "theoretical_basis", ""),
            "metrics": {
                "hf_like_score": float(metrics.get("hf_like_score", float("inf")))
                if metrics
                else None,
                "validation_objective": float(metrics.get("validation_objective", float("inf")))
                if metrics
                else None,
                "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
                if metrics
                else None,
                "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            },
        }
        parent_blobs.append(blob)

    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt, llm_op="E2", prompt_path=e2_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def paradigm_shift_free_loss(
    prompt_path: str,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        parent_blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "pseudocode": parent.pseudocode,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "code": parent.code,
                "theoretical_basis": getattr(parent, "theoretical_basis", ""),
                "metrics": {
                    "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
                    "validation_objective": float(metrics.get("validation_objective", float("inf")))
                    if metrics
                    else None,
                    "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
                    "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
                    "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
                },
            }
        )
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op="LOSS_PARADIGM_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def structure_shift_free_loss(
    prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op="LOSS_STRUCTURE_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def constraint_inject_free_loss(
    prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    prompt = _read_prompt(prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    raw = _call_llm(prompt, llm_op="LOSS_CONSTRAINT_INJECT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def m2_tune_hparams(
    m2_prompt_path: str,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """M2: hyperparameter-only tuning; structurally identical to parent."""

    prompt = _read_prompt(m2_prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0))
            if metrics
            else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt, llm_op="M2", prompt_path=m2_prompt_path)
    json_str = _extract_json_object(raw)
    tuned = parse_free_loss_from_text(json_str)

    parent_hp = dict(parent.hyperparams or {})
    tuned_hp = dict(tuned.hyperparams or {})

    # Enforce "hyperparams-only": keep structure and restrict hyperparam keys.
    if parent.code.strip():
        # If the parent has explicit code, only keep keys that already exist,
        # since new keys won't be used unless the code changes (which is forbidden in M2).
        tuned_hp = {k: tuned_hp.get(k, parent_hp.get(k)) for k in parent_hp.keys()}
    else:
        # Template-based losses may rely on compiler-known hyperparams; keep tuned as-is.
        tuned_hp = tuned_hp or parent_hp

    return FreeLossIR(
        name=tuned.name or f"{parent.name}_m2",
        intuition=tuned.intuition or parent.intuition,
        pseudocode=tuned.pseudocode or parent.pseudocode,
        hyperparams=tuned_hp,
        operators_used=list(parent.operators_used),
        implementation_hint=parent.implementation_hint,
        code=parent.code,
        theoretical_basis=tuned.theoretical_basis or getattr(parent, "theoretical_basis", ""),
    )


def m3_simplify_loss(
    m3_prompt_path: str,
    candidate: FreeLossIR,
    failure_reason: Mapping[str, Any],
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """M3: simplify/stabilize a candidate loss, given a failure reason."""

    prompt = _read_prompt(m3_prompt_path)
    payload = {
        "candidate": {
            "name": candidate.name,
            "intuition": candidate.intuition,
            "pseudocode": candidate.pseudocode,
            "hyperparams": candidate.hyperparams,
            "operators_used": candidate.operators_used,
            "code": candidate.code,
            "theoretical_basis": getattr(candidate, "theoretical_basis", ""),
        },
        "failure_reason": dict(failure_reason),
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        feedback_blob = json.dumps(global_feedback, indent=2, ensure_ascii=False)
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + feedback_blob

    raw = _call_llm(prompt, llm_op="M3", prompt_path=m3_prompt_path)
    json_str = _extract_json_object(raw)
    simplified = parse_free_loss_from_text(json_str)

    # Preserve the contract required by gates, even if the model drifts.
    simplified.implementation_hint = candidate.implementation_hint
    return simplified


def repair_free_loss(
    repair_prompt_path: str,
    failed_ir: FreeLossIR,
    failure_reason: Mapping[str, Any],
) -> FreeLossIR:
    prompt = _read_prompt(repair_prompt_path)
    payload = {
        "candidate": {
            "name": failed_ir.name,
            "intuition": failed_ir.intuition,
            "pseudocode": failed_ir.pseudocode,
            "hyperparams": failed_ir.hyperparams,
            "operators_used": failed_ir.operators_used,
            "implementation_hint": asdict(failed_ir.implementation_hint),
            "code": failed_ir.code,
            "theoretical_basis": getattr(failed_ir, "theoretical_basis", ""),
        },
        "failure_reason": failure_reason,
    }
    prompt = prompt + "\n\nCANDIDATE_AND_FAILURE_JSON:\n" + json.dumps(payload, indent=2)
    raw = _call_llm(prompt, llm_op="REPAIR", prompt_path=repair_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def repair_expects_with_prompt(
    expects_repair_prompt_path: str,
    ir: FreeLossIR,
) -> FreeLossIR:
    """Use a lightweight LLM prompt to normalize implementation_hint.expects.

    This is only used when we already have an expects list, to coerce it into
    a clean list of short input names.
    """

    prompt = _read_prompt(expects_repair_prompt_path)
    payload = asdict(ir)
    prompt = prompt + "\n\nIR_JSON:\n" + json.dumps(payload, indent=2)
    raw = _call_llm(prompt, llm_op="EXPECTS_REPAIR", prompt_path=expects_repair_prompt_path)
    json_str = _extract_json_object(raw)
    return parse_free_loss_from_text(json_str)


def repair_from_gate_failure(
    directed_repair_prompt_path: str,
    parent_ir: FreeLossIR,
    *,
    strategy: str,
    gate_spec: Mapping[str, Any],
    fail_report: Mapping[str, Any],
    counterexamples: Sequence[Mapping[str, Any]],
    allowed_keys: Sequence[str],
    global_feedback: Mapping[str, Any] | None = None,
) -> FreeLossIR:
    """Generate a repaired child candidate guided by gate diagnostics.

    The prompt is designed for CEGIS-style repair: provide a failure report
    plus counterexamples (visible tests) and request a structured patch.
    """

    strategy = str(strategy or "").strip().lower()
    if strategy not in {"e1", "e2", "m1", "m2"}:
        raise ValueError(f"Unknown directed repair strategy: {strategy!r}")

    prompt = _read_prompt(directed_repair_prompt_path)
    prompt = (
        prompt
        + "\n\nSTRATEGY:\n"
        + strategy
        + "\n\nPARENT_CODE:\n"
        + (parent_ir.code or "").strip()
        + "\n\nGATE_SPEC_JSON:\n"
        + json.dumps(dict(gate_spec), indent=2, ensure_ascii=False)
        + "\n\nFAIL_REPORT_JSON:\n"
        + json.dumps(dict(fail_report), indent=2, ensure_ascii=False)
        + "\n\nCOUNTEREXAMPLES_JSON:\n"
        + json.dumps(list(counterexamples), indent=2, ensure_ascii=False)
        + "\n\nCONTRACT_JSON:\n"
        + json.dumps(
            {
                "allowed_keys": list(allowed_keys),
                "required_function": "generated_loss(batch, model_output, extra)",
                "no_imports": True,
                "no_external_state": True,
                "must_be_numerically_stable": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)

    raw = _call_llm(prompt, llm_op=f"DIR_REPAIR_{strategy}", prompt_path=directed_repair_prompt_path)
    obj = json.loads(_extract_json_object(raw))

    out_strategy = str(obj.get("strategy", strategy) or strategy).strip().lower()
    expects_raw = obj.get("expects", None)
    code = str(obj.get("code", "")).strip()
    reasoning = str(obj.get("reasoning_brief", "")).strip()

    if out_strategy not in {"e1", "e2", "m1", "m2"}:
        out_strategy = strategy

    if not code:
        raise ValueError("Directed repair output missing 'code'.")
    if "def generated_loss" not in code:
        raise ValueError("Directed repair code must define 'generated_loss'.")

    expects: list[str]
    if isinstance(expects_raw, (list, tuple)):
        expects = [str(x) for x in expects_raw]
    elif expects_raw is None:
        expects = [str(x) for x in (parent_ir.implementation_hint.expects or [])]
    else:
        expects = [str(expects_raw)]

    # Optional: allow the model to update these, but default to the parent.
    name = str(obj.get("name", "")).strip() or f"{parent_ir.name}_dr_{out_strategy}"
    intuition = str(obj.get("intuition", "")).strip() or parent_ir.intuition
    if reasoning:
        intuition = f"{intuition}\nDirected repair ({out_strategy}): {reasoning}".strip()
    pseudocode = str(obj.get("pseudocode", "")).strip() or parent_ir.pseudocode
    hyperparams = obj.get("hyperparams", None)
    if not isinstance(hyperparams, dict):
        hyperparams = dict(parent_ir.hyperparams or {})
    operators_used = obj.get("operators_used", None)
    if isinstance(operators_used, (list, tuple)):
        operators_list = [str(x) for x in operators_used] or list(parent_ir.operators_used)
    else:
        operators_list = list(parent_ir.operators_used)

    mode = str(obj.get("mode", "") or parent_ir.implementation_hint.mode or "pairwise").strip().lower()
    if mode not in {"pairwise", "setwise"}:
        mode = str(parent_ir.implementation_hint.mode or "pairwise").strip().lower() or "pairwise"

    return FreeLossIR(
        name=name,
        intuition=intuition,
        pseudocode=pseudocode,
        hyperparams=dict(hyperparams),
        operators_used=operators_list,
        implementation_hint=type(parent_ir.implementation_hint)(
            expects=expects,
            returns="scalar",
            mode=mode,
        ),
        code=code,
        theoretical_basis=str(obj.get("theoretical_basis", "")).strip()
        or getattr(parent_ir, "theoretical_basis", ""),
    )


def paradigm_shift_free_loss_with_meta(
    prompt_path: str,
    *,
    parents: Sequence[FreeLossIR],
    parents_fitness: Sequence[Mapping[str, Any]] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> tuple[FreeLossIR, Mapping[str, Any]]:
    prompt = _read_prompt(prompt_path)
    parent_blobs = []
    for idx, parent in enumerate(parents):
        metrics: Mapping[str, Any] = {}
        if parents_fitness is not None and idx < len(parents_fitness):
            metrics = parents_fitness[idx]
        parent_blobs.append(
            {
                "index": idx,
                "name": parent.name,
                "intuition": parent.intuition,
                "pseudocode": parent.pseudocode,
                "hyperparams": parent.hyperparams,
                "operators_used": parent.operators_used,
                "code": parent.code,
                "theoretical_basis": getattr(parent, "theoretical_basis", ""),
                "metrics": {
                    "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
                    "validation_objective": float(metrics.get("validation_objective", float("inf")))
                    if metrics
                    else None,
                    "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
                    "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
                    "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
                },
            }
        )
    prompt = prompt + "\n\nPARENTS_JSON:\n" + json.dumps(parent_blobs, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    prompt_sha1 = sha1(prompt.encode("utf-8")).hexdigest()
    raw = _call_llm(prompt, llm_op="LOSS_PARADIGM_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return (
        parse_free_loss_from_text(json_str),
        {
            "llm_op": "LOSS_PARADIGM_SHIFT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def structure_shift_free_loss_with_meta(
    prompt_path: str,
    *,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> tuple[FreeLossIR, Mapping[str, Any]]:
    prompt = _read_prompt(prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    prompt_sha1 = sha1(prompt.encode("utf-8")).hexdigest()
    raw = _call_llm(prompt, llm_op="LOSS_STRUCTURE_SHIFT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return (
        parse_free_loss_from_text(json_str),
        {
            "llm_op": "LOSS_STRUCTURE_SHIFT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def constraint_inject_free_loss_with_meta(
    prompt_path: str,
    *,
    parent: FreeLossIR,
    parent_fitness: Mapping[str, Any] | None = None,
    global_feedback: Mapping[str, Any] | None = None,
) -> tuple[FreeLossIR, Mapping[str, Any]]:
    prompt = _read_prompt(prompt_path)
    metrics: Mapping[str, Any] = parent_fitness or {}
    parent_blob = {
        "name": parent.name,
        "intuition": parent.intuition,
        "pseudocode": parent.pseudocode,
        "hyperparams": parent.hyperparams,
        "operators_used": parent.operators_used,
        "code": parent.code,
        "theoretical_basis": getattr(parent, "theoretical_basis", ""),
        "metrics": {
            "hf_like_score": float(metrics.get("hf_like_score", float("inf"))) if metrics else None,
            "validation_objective": float(metrics.get("validation_objective", float("inf")))
            if metrics
            else None,
            "generalization_penalty": float(metrics.get("generalization_penalty", 0.0)) if metrics else None,
            "pair_count": int(metrics.get("pair_count", 0) or 0) if metrics else 0,
            "fitness": float(metrics.get("fitness", float("inf"))) if metrics else None,
        },
    }
    prompt = prompt + "\n\nPARENT_JSON:\n" + json.dumps(parent_blob, indent=2, ensure_ascii=False)
    if global_feedback is not None:
        prompt = prompt + "\n\nGLOBAL_FEEDBACK_JSON:\n" + json.dumps(global_feedback, indent=2, ensure_ascii=False)
    prompt_sha1 = sha1(prompt.encode("utf-8")).hexdigest()
    raw = _call_llm(prompt, llm_op="LOSS_CONSTRAINT_INJECT", prompt_path=prompt_path)
    json_str = _extract_json_object(raw)
    return (
        parse_free_loss_from_text(json_str),
        {
            "llm_op": "LOSS_CONSTRAINT_INJECT",
            "prompt_path": str(prompt_path),
            "prompt_sha1": str(prompt_sha1),
        },
    )


def compile_free_loss_candidate(
    ir: FreeLossIR,
    *,
    operator_whitelist: Sequence[str],
) -> CompiledFreeLoss:
    return compile_free_loss(ir, operator_whitelist=operator_whitelist)
