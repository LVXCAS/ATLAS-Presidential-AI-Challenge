#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = str(value).strip().lower()
    if not v:
        return True
    if v in {"***", "****", "*****"}:
        return True
    tokens = ("your_", "your-", "changeme", "replace", "example", "password_here", "token_here", "key_here")
    return any(token in v for token in tokens)


def _load_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in env:
            if _is_placeholder(env.get(key)) and not _is_placeholder(value):
                env[key] = value
        else:
            env[key] = value
    return env


def _get_env(name: str, env: Dict[str, str]) -> Optional[str]:
    return os.environ.get(name) or env.get(name)


def _build_llm_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _call_llm(api_base: str, api_key: str, payload: Dict) -> str:
    request = urllib.request.Request(
        _build_llm_url(api_base),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response missing choices.")
    return choices[0].get("message", {}).get("content", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional LLM summary for evaluation output.")
    parser.add_argument(
        "--input",
        default="submission/evaluation_results.json",
        help="Path to evaluation results JSON.",
    )
    parser.add_argument("--output", default=None, help="Optional output file for summary.")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=300, help="LLM max tokens.")
    parser.add_argument("--enable-live", action="store_true", help="Required to allow API calls.")
    args = parser.parse_args()

    root = _repo_root()
    dotenv = _load_dotenv(root / ".env")
    live_enabled = args.enable_live or _truthy(_get_env("USE_LIVE_DATA", dotenv))
    if not live_enabled:
        print(
            "Live LLM access is disabled. Re-run with --enable-live or set USE_LIVE_DATA=true in .env.",
            file=sys.stderr,
        )
        return 1

    api_base = _get_env("LLM_API_BASE", dotenv) or ("https://api.deepseek.com" if _get_env("DEEPSEEK_API_KEY", dotenv) else "")
    api_key = _get_env("LLM_API_KEY", dotenv) or _get_env("DEEPSEEK_API_KEY", dotenv)
    if not api_key or not api_base:
        print("Missing LLM_API_KEY/LLM_API_BASE or DEEPSEEK_API_KEY in environment or .env.", file=sys.stderr)
        return 2

    input_path = root / args.input
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    payload_data = json.loads(input_path.read_text())
    prompt = (
        "You are an educational risk-literacy assistant. Summarize the evaluation results "
        "without giving trading advice or price predictions. Explain risk posture changes "
        "in plain, student-friendly language and cite the main risk flags."
        f"\n\nEvaluation JSON:\n{json.dumps(payload_data, indent=2)}"
    )

    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You write concise, safe, student-friendly summaries."},
            {"role": "user", "content": prompt},
        ],
        "temperature": max(0.0, args.temperature),
        "max_tokens": max(50, args.max_tokens),
    }

    try:
        summary = _call_llm(api_base, api_key, body)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return 3

    if args.output:
        output_path = root / args.output
        output_path.write_text(summary)
        print(f"Wrote summary to {output_path}")
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
