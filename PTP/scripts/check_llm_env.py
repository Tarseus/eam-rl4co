from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable


def _repo_root() -> Path:
    # This file lives at <repo_root>/PTP/scripts/check_llm_env.py
    return Path(__file__).resolve().parents[2]


def _candidate_env_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = (os.getenv("OPENAI_DOTENV_PATH") or "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser().resolve())
    paths.append((Path.cwd() / ".env").resolve())
    paths.append((_repo_root() / ".env").resolve())

    unique: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def _mask_secret(value: str) -> str:
    text = str(value or "")
    if len(text) <= 10:
        return "*" * len(text)
    return f"{text[:6]}...{text[-4:]}"


def _load_dotenv_fallback(path: Path) -> int:
    loaded = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded

    for raw_line in lines:
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
    return loaded


def _print_paths(paths: Iterable[Path]) -> None:
    print("dotenv search paths:")
    for p in paths:
        exists = "exists" if p.is_file() else "missing"
        print(f"  - {p} [{exists}]")


def _load_env(paths: list[Path]) -> tuple[str | None, str]:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None  # type: ignore[assignment]

    if load_dotenv is not None:
        for p in paths:
            if p.is_file():
                load_dotenv(dotenv_path=str(p), override=False)
                return str(p), "python-dotenv"
        return None, "python-dotenv"

    for p in paths:
        if p.is_file():
            _load_dotenv_fallback(p)
            return str(p), "fallback-parser"
    return None, "fallback-parser"


def _ping_openai(timeout_s: float) -> tuple[bool, str]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        return False, f"openai import failed: {exc}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY is missing"
    base_url = _normalize_openai_base_url(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True, "chat.completions request succeeded"
    except Exception as exc:
        return False, f"chat.completions request failed: {exc}"


def _normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = str(raw_base_url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if base_url.endswith(suffix):
        base_url = base_url[: -len(suffix)].rstrip("/")
    return base_url or str(raw_base_url)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose LLM environment loading.")
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Send a minimal chat.completions request to verify API connectivity.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for --ping request.",
    )
    args = parser.parse_args()

    print(f"python: {sys.version.split()[0]}")
    print(f"cwd: {Path.cwd()}")
    print(f"repo_root: {_repo_root()}")

    paths = _candidate_env_paths()
    _print_paths(paths)
    loaded_from, loader = _load_env(paths)
    print(f"loader: {loader}")
    print(f"loaded_from: {loaded_from or '<none>'}")

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {'set' if api_key else 'missing'}")
    if api_key:
        print(f"OPENAI_API_KEY(masked): {_mask_secret(api_key)}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', '<default>')}")
    print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', '<default>')}")

    ok = bool(api_key)
    if args.ping:
        ping_ok, detail = _ping_openai(timeout_s=float(args.timeout))
        ok = ok and ping_ok
        print(f"ping: {'ok' if ping_ok else 'failed'}")
        print(f"ping_detail: {detail}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

