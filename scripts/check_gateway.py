#!/usr/bin/env python3
"""Check a local OpenAI-compatible gateway (OmniRoute/Ollama/LM Studio): list the
models it serves and test which are actually reachable.

Usage:
  python scripts/check_gateway.py                  # summarize families + sample-probe
  python scripts/check_gateway.py --model <id>     # test one model id
  python scripts/check_gateway.py --probe <prefix> # probe every model with that prefix
                                                   #   (e.g. --probe cfp, --probe kilo-auto)
  python scripts/check_gateway.py --all            # probe every model (slow for big catalogs)

Config: reads OMNIRoute_BASE_URL / OMNIRoute_API_KEY from .env (or env vars, or
--base-url / --api-key arguments).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter, OrderedDict
from pathlib import Path


def safe(s: str, n: int = 160) -> str:
    return "".join(c if ord(c) < 128 else "?" for c in str(s or ""))[:n]


def load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


class GatewayChecker:
    def __init__(self, base_url: str, api_key: str, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key or 'local'}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def _get(self, path: str, timeout: int = 30) -> dict:
        req = urllib.request.Request(self.base_url + path, headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def list_models(self) -> list[dict]:
        return self._get("/models").get("data", [])

    async def test_model(self, model: str) -> tuple[int, str]:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 8,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, headers=self.headers, method="POST"
        )

        def _run() -> tuple[int, str]:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return 200, safe(content)
            except urllib.error.HTTPError as e:
                try:
                    err = safe(json.loads(e.read().decode()).get("error", {}).get("message", ""), 200)
                except Exception:
                    err = ""
                return e.code, err
            except Exception as e:  # noqa: BLE001
                return -1, safe(repr(e), 200)

        return await asyncio.to_thread(_run)

    async def probe(self, model_ids: list[str], concurrency: int = 6) -> list[tuple[str, int, str]]:
        sem = asyncio.Semaphore(concurrency)

        async def one(mid: str) -> tuple[str, int, str]:
            async with sem:
                code, msg = await self.test_model(mid)
                return mid, code, msg

        out: list[tuple[str, int, str]] = []
        for i in range(0, len(model_ids), concurrency):
            batch = model_ids[i:i + concurrency]
            out.extend(await asyncio.gather(*(one(m) for m in batch)))
        return out


async def run(args: argparse.Namespace) -> int:
    env = load_env(Path(__file__).resolve().parent.parent / ".env")
    base_url = args.base_url or os.environ.get("OMNIRoute_BASE_URL") or env.get("OMNIRoute_BASE_URL")
    api_key = args.api_key or os.environ.get("OMNIRoute_API_KEY") or env.get("OMNIRoute_API_KEY")
    if not base_url:
        print("error: no gateway base url (set OMNIRoute_BASE_URL in .env or pass --base-url)")
        return 2

    checker = GatewayChecker(base_url, api_key or "local")
    print(f"gateway: {base_url}")
    try:
        models = checker.list_models()
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot reach {base_url}/models: {exc}")
        return 1

    ids = [m.get("id", "") for m in models if m.get("id")]
    print(f"models served: {len(ids)}\n")

    if args.model:
        targets = [args.model]
    elif args.probe:
        targets = [i for i in ids if i.startswith(args.probe)]
        print(f"== family '{args.probe}': {len(targets)} models ==")
    elif args.all:
        targets = ids
    else:
        fam = Counter(i.split("/")[0] for i in ids)
        print("== model families ==")
        for f, c in fam.most_common():
            print(f"{c:5d}  {f}")
        print("\n== sample probing (first model of each family) ==")
        samples: "OrderedDict[str, str]" = OrderedDict()
        for i in ids:
            samples.setdefault(i.split("/")[0], i)
        targets = list(samples.values())

    if not targets:
        print("no models matched")
        return 0

    print()
    results = await checker.probe(targets)
    working: list[tuple[str, int, str]] = []
    for mid, code, msg in results:
        tag = "OK " if code == 200 else f"{code}"
        print(f"[{tag}] {mid}")
        if code == 200:
            print(f"    -> {msg!r}")
            working.append((mid, code, msg))
        elif msg:
            print(f"    -> {msg}")
        sys.stdout.flush()

    print("\n=== WORKING MODELS ===")
    for mid, _code, msg in working:
        print(f"{mid} -> {msg!r}")
    if not working:
        print("(none - the gateway's upstream connections may need re-authentication "
              "in the OmniRoute dashboard)")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", help="gateway base url (default: OMNIRoute_BASE_URL from .env)")
    p.add_argument("--api-key", help="gateway api key (default: OMNIRoute_API_KEY from .env)")
    p.add_argument("--model", help="test a single model id")
    p.add_argument("--probe", help="probe all models with this id prefix (e.g. cfp, kilo-auto)")
    p.add_argument("--all", action="store_true", help="probe every model (slow)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()