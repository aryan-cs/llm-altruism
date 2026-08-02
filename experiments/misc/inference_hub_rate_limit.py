"""Cross-process rate limiting for NVIDIA InferenceHub requests.

Every hosted caller in this repository goes through :class:`InferenceHubClient`,
which uses this module before opening a network connection.  The scheduler is
shared by threads and processes through a small file locked with ``flock``.  It
limits both starts per second and simultaneous requests, globally and per
upstream provider, while allowing different providers to make bounded progress
in parallel.
"""

from __future__ import annotations

import email.utils
import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


RATE_LIMIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RateLimitPolicy:
    """Conservative fixed policy shared by all InferenceHub workflows."""

    global_concurrency: int = 4
    provider_concurrency: int = 1
    global_requests_per_second: float = 2.0
    provider_requests_per_second: float = 0.5
    lease_seconds: float = 900.0
    poll_seconds: float = 0.05
    transient_cooldown_seconds: float = 5.0
    throttle_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        integer_fields = (self.global_concurrency, self.provider_concurrency)
        positive_fields = (
            self.global_requests_per_second,
            self.provider_requests_per_second,
            self.lease_seconds,
            self.poll_seconds,
            self.transient_cooldown_seconds,
            self.throttle_cooldown_seconds,
        )
        if any(isinstance(value, bool) or value < 1 for value in integer_fields):
            raise ValueError("Rate-limit concurrency values must be positive integers.")
        if any(value <= 0 for value in positive_fields):
            raise ValueError("Rate-limit timing values must be positive.")
        if self.provider_concurrency > self.global_concurrency:
            raise ValueError("Provider concurrency cannot exceed global concurrency.")

    def evidence(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = RATE_LIMIT_SCHEMA_VERSION
        payload["algorithm"] = "cross_process_provider_aware_leaky_bucket_with_leases"
        payload["policy_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


def provider_for_route(route: object) -> str:
    """Map an exact route to the upstream provider used for throttling."""

    if not isinstance(route, str) or not route.strip():
        return "inference_hub_metadata"
    normalized = route.strip().lower()
    pieces = [piece for piece in normalized.split("/") if piece]
    if not pieces:
        return "inference_hub_metadata"
    known_upstreams = {
        "anthropic",
        "cerebras",
        "cohere",
        "deepseek",
        "deepseek-ai",
        "google",
        "meta",
        "microsoft",
        "minimax",
        "minimaxai",
        "mistral",
        "mistralai",
        "moonshot",
        "moonshotai",
        "nvidia",
        "openai",
        "qwen",
        "xai",
        "zai-org",
    }
    if pieces[0] in {"aws", "azure", "bedrock", "gcp", "nvidia", "us"}:
        for piece in pieces[1:]:
            if piece in known_upstreams:
                pieces = [piece]
                break
    return pieces[0]


def _default_state_path(scope_id: str) -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    scope_hash = hashlib.sha256(scope_id.encode("utf-8")).hexdigest()[:24]
    return Path(tempfile.gettempdir()) / (
        f"llm-altruism-inference-hub-rate-limit-{uid}-{scope_hash}.json"
    )


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InferenceHubRateLimiter:
    """A crash-tolerant, file-backed limiter shared across local processes."""

    def __init__(
        self,
        *,
        policy: RateLimitPolicy | None = None,
        state_path: Path | None = None,
        scope_id: str = "default",
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.policy = policy or RateLimitPolicy()
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("Rate-limit scope_id must be a non-empty string.")
        self.state_path = (state_path or _default_state_path(scope_id)).resolve()
        self.lock_path = self.state_path.with_name(f".{self.state_path.name}.lock")
        self._clock = clock
        self._sleep = sleep

    def contract(self) -> dict[str, Any]:
        return self.policy.evidence()

    @contextmanager
    def _locked_state(self) -> Iterator[dict[str, Any]]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            if os.name == "posix":
                os.chmod(self.lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = self._read_state()
            try:
                yield state
                self._write_state(state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _empty_state(self) -> dict[str, Any]:
        return {
            "schema_version": RATE_LIMIT_SCHEMA_VERSION,
            "policy_sha256": self.policy.evidence()["policy_sha256"],
            "next_global_at": 0.0,
            "next_provider_at": {},
            "leases": [],
        }

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "InferenceHub rate-limit state is unreadable; refusing to fail open."
            ) from error
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != RATE_LIMIT_SCHEMA_VERSION
            or state.get("policy_sha256")
            != self.policy.evidence()["policy_sha256"]
            or not isinstance(state.get("next_provider_at"), dict)
            or not isinstance(state.get("leases"), list)
        ):
            raise RuntimeError(
                "InferenceHub rate-limit state/policy mismatch; refusing to fail open."
            )
        return state

    def _write_state(self, state: Mapping[str, Any]) -> None:
        descriptor, temporary = tempfile.mkstemp(
            dir=self.state_path.parent,
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            if os.name == "posix":
                os.chmod(self.state_path, 0o600)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _prune(self, state: dict[str, Any], now: float) -> None:
        state["leases"] = [
            lease
            for lease in state["leases"]
            if isinstance(lease, dict)
            and isinstance(lease.get("expires_at"), (int, float))
            and float(lease["expires_at"]) > now
            and _pid_alive(lease.get("pid"))
        ]

    def acquire(self, provider: str) -> str:
        normalized = provider_for_route(provider)
        while True:
            now = self._clock()
            wait_for = self.policy.poll_seconds
            token: str | None = None
            with self._locked_state() as state:
                self._prune(state, now)
                leases = state["leases"]
                provider_leases = [
                    lease for lease in leases if lease.get("provider") == normalized
                ]
                next_global = float(state.get("next_global_at", 0.0))
                next_provider = float(state["next_provider_at"].get(normalized, 0.0))
                slot_at = max(now, next_global, next_provider)
                capacity = (
                    len(leases) < self.policy.global_concurrency
                    and len(provider_leases) < self.policy.provider_concurrency
                )
                if capacity and slot_at <= now:
                    token = f"rl_{uuid.uuid4().hex}"
                    leases.append(
                        {
                            "token": token,
                            "pid": os.getpid(),
                            "provider": normalized,
                            "acquired_at": now,
                            "expires_at": now + self.policy.lease_seconds,
                        }
                    )
                    state["next_global_at"] = now + (
                        1.0 / self.policy.global_requests_per_second
                    )
                    state["next_provider_at"][normalized] = now + (
                        1.0 / self.policy.provider_requests_per_second
                    )
                elif capacity:
                    wait_for = max(self.policy.poll_seconds, slot_at - now)
            if token is not None:
                return token
            self._sleep(wait_for)

    def release(self, token: str) -> None:
        with self._locked_state() as state:
            now = self._clock()
            self._prune(state, now)
            state["leases"] = [
                lease for lease in state["leases"] if lease.get("token") != token
            ]

    def renew(self, token: str) -> bool:
        """Extend one live owned lease; return false if ownership was lost."""

        with self._locked_state() as state:
            now = self._clock()
            self._prune(state, now)
            matches = [
                lease
                for lease in state["leases"]
                if lease.get("token") == token and lease.get("pid") == os.getpid()
            ]
            if len(matches) != 1:
                return False
            matches[0]["expires_at"] = now + self.policy.lease_seconds
            matches[0]["renewed_at"] = now
            return True

    @contextmanager
    def limit(self, provider: str) -> Iterator[None]:
        token = self.acquire(provider)
        stopped = threading.Event()
        lost = threading.Event()

        def heartbeat() -> None:
            interval = max(
                self.policy.poll_seconds,
                self.policy.lease_seconds / 3.0,
            )
            while not stopped.wait(interval):
                if not self.renew(token):
                    lost.set()
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name="inference-hub-rate-limit-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield
            if lost.is_set():
                raise RuntimeError(
                    "InferenceHub rate-limit lease ownership was lost during request."
                )
        finally:
            stopped.set()
            heartbeat_thread.join(timeout=max(1.0, self.policy.poll_seconds * 2.0))
            self.release(token)

    def penalize(
        self,
        provider: str,
        *,
        http_status: int,
        retry_after: str | None = None,
    ) -> float:
        """Publish a shared cooldown after a throttle or transient server error."""

        normalized = provider_for_route(provider)
        now = self._clock()
        default = (
            self.policy.throttle_cooldown_seconds
            if http_status in {429, 529}
            else self.policy.transient_cooldown_seconds
        )
        delay = max(default, self._parse_retry_after(retry_after, now))
        with self._locked_state() as state:
            state["next_global_at"] = max(
                float(state.get("next_global_at", 0.0)), now + min(delay, default)
            )
            current = float(state["next_provider_at"].get(normalized, 0.0))
            state["next_provider_at"][normalized] = max(current, now + delay)
        return delay

    @staticmethod
    def _parse_retry_after(value: str | None, now: float) -> float:
        if not value:
            return 0.0
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value).timestamp()
            except (TypeError, ValueError, OverflowError):
                return 0.0
            return max(0.0, parsed - now)
