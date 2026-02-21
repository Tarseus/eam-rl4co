from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


LOGGER = logging.getLogger("ptp_discovery.pref_builder_ir")


@dataclass
class PreferenceBuilderImplementationHint:
    expects: Sequence[str]
    returns: str
    mode: str = "pairwise"


@dataclass
class PreferenceBuilderIR:
    """Intermediate representation for a preference builder.

    The builder is expected to provide concrete Python code in `code` defining:

        generated_builder(feature_cache, extra) -> PrefBatch
    """

    name: str
    intuition: str
    implementation_hint: PreferenceBuilderImplementationHint
    code: str = ""


def ir_from_json(obj: Mapping[str, Any]) -> PreferenceBuilderIR:
    name = str(obj.get("name", "")).strip()
    intuition = str(obj.get("intuition", "")).strip()
    code = str(obj.get("code", "")).strip()

    impl_raw = obj.get("implementation_hint", {}) or {}
    if not isinstance(impl_raw, Mapping):
        LOGGER.debug("implementation_hint not object; raw=%r", impl_raw)
        impl_raw = {}

    expects_raw = impl_raw.get("expects", []) or []
    if isinstance(expects_raw, (list, tuple)):
        expects = [str(x) for x in expects_raw]
    elif expects_raw is None:
        expects = []
    elif isinstance(expects_raw, Mapping):
        expects = [str(k) for k in expects_raw.keys()]
    else:
        LOGGER.debug(
            "implementation_hint.expects not array or Mapping; type=%s, raw=%r",
            type(expects_raw),
            expects_raw,
        )
        expects = [str(expects_raw)]

    returns = str(impl_raw.get("returns", "")).strip()
    mode_raw = str(impl_raw.get("mode", "")).strip().lower()
    if mode_raw not in {"", "pairwise", "setwise", "listwise"}:
        LOGGER.debug("implementation_hint.mode invalid; raw=%r", mode_raw)
        mode_raw = ""

    impl = PreferenceBuilderImplementationHint(
        expects=expects,
        returns=returns or "PrefBatch",
        mode=mode_raw or "pairwise",
    )

    return PreferenceBuilderIR(
        name=name or "unnamed_preference_builder",
        intuition=intuition,
        implementation_hint=impl,
        code=code,
    )

