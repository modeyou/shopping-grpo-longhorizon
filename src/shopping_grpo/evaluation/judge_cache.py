"""Content-addressed cache for stable cross-model Judge reuse."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
import time

from shopping_grpo.evaluation.artifacts import load_json, write_json_atomic
from shopping_grpo.evaluation.manifest import canonical_json_sha256


JUDGE_CACHE_VERSION = "shopping-semantic-judge-cache-v1"


class JudgeCacheError(ValueError):
    """Raised when a shared Judge cache entry violates its identity."""


def semantic_request_spec(
    *,
    model: str,
    base_url: str,
    prompt_version: str,
    max_tokens: int,
    thinking: bool,
    messages: list[Mapping],
) -> dict:
    """Build the complete non-secret identity of one semantic Judge call."""

    return {
        "model": str(model),
        "base_url": str(base_url).rstrip("/"),
        "prompt_version": str(prompt_version),
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking": bool(thinking),
        "response_format": "json_object",
        "messages": deepcopy(messages),
    }


class SemanticJudgeCache:
    """Share one validated Judge response for each exact semantic request."""

    def __init__(self, root: str | Path, *, wait_timeout: float = 600.0):
        self.root = Path(root)
        self.wait_timeout = float(wait_timeout)
        if self.wait_timeout <= 0:
            raise ValueError("Judge cache wait_timeout must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, request_sha256: str) -> tuple[Path, Path]:
        if len(request_sha256) != 64:
            raise JudgeCacheError("semantic request hash must be SHA-256")
        return (
            self.root / f"{request_sha256}.json",
            self.root / f"{request_sha256}.lock",
        )

    @staticmethod
    def _validate_entry(entry: Mapping, *, request_spec: Mapping) -> dict:
        expected_sha256 = canonical_json_sha256(request_spec)
        if entry.get("schema_version") != JUDGE_CACHE_VERSION:
            raise JudgeCacheError("unsupported semantic Judge cache schema")
        if entry.get("semantic_request_sha256") != expected_sha256:
            raise JudgeCacheError("semantic Judge cache hash mismatch")
        if entry.get("request_spec") != request_spec:
            raise JudgeCacheError("semantic Judge cache request identity mismatch")
        response = entry.get("response")
        if not isinstance(response, Mapping):
            raise JudgeCacheError("semantic Judge cache response must be an object")
        return deepcopy(dict(response))

    def get_or_compute(
        self,
        request_spec: Mapping,
        compute: Callable[[], Mapping],
    ) -> tuple[dict, bool]:
        """Return a cached response or atomically publish one computed response."""

        frozen_spec = deepcopy(dict(request_spec))
        request_sha256 = canonical_json_sha256(frozen_spec)
        entry_path, lock_path = self._paths(request_sha256)
        if entry_path.exists():
            return self._validate_entry(
                load_json(entry_path), request_spec=frozen_spec
            ), True

        try:
            lock_path.mkdir()
            owns_lock = True
        except FileExistsError:
            owns_lock = False

        if not owns_lock:
            deadline = time.monotonic() + self.wait_timeout
            while time.monotonic() < deadline:
                if entry_path.exists():
                    return self._validate_entry(
                        load_json(entry_path), request_spec=frozen_spec
                    ), True
                time.sleep(0.1)
            raise JudgeCacheError(
                f"timed out waiting for semantic Judge cache {request_sha256}; "
                f"inspect stale lock {lock_path}"
            )

        try:
            response = compute()
            if not isinstance(response, Mapping):
                raise JudgeCacheError("Judge cache compute result must be an object")
            entry = {
                "schema_version": JUDGE_CACHE_VERSION,
                "semantic_request_sha256": request_sha256,
                "request_spec": frozen_spec,
                "response": deepcopy(dict(response)),
            }
            write_json_atomic(entry_path, entry)
            return deepcopy(dict(response)), False
        finally:
            lock_path.rmdir()
