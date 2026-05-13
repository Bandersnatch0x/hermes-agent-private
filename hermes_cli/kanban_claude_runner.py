"""Kanban worker runner that delegates the task body to the ``claude`` CLI.

Spawned by the kanban dispatcher (``kanban_db.py:_default_spawn``) when a
profile's ``runner.kind`` is set to ``claude-code``. This module is the
``claude-code`` equivalent of "spawn ``hermes -p <profile> chat -q ...``"
— it runs ``claude -p`` in print mode, parses the structured JSON result,
and writes the outcome back through the standard ``hermes kanban`` CLI
(``show`` / ``complete`` / ``block`` / ``heartbeat``).

Run via ``python -m hermes_cli.kanban_claude_runner``. All inputs come
from the worker environment the dispatcher sets up:

  HERMES_KANBAN_TASK       task id
  HERMES_KANBAN_WORKSPACE  workspace dir to ``cwd`` into
  HERMES_KANBAN_BOARD      kanban board slug (passed through to ``hermes kanban``)
  HERMES_PROFILE           profile name (used to look up ``runner:`` config)
  HERMES_HOME              profile-scoped home (set by the dispatcher)

This script is fire-and-forget from the dispatcher's perspective: it owns
its own heartbeat thread and writes the final ``complete``/``block``
itself so the kanban claim lifecycle stays consistent with the
``hermes``-runner path.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


# < TTL/2 (TTL is 15 min in kanban_db.DEFAULT_CLAIM_TTL_SECONDS) so a single
# missed heartbeat still leaves us inside the lock window.
HEARTBEAT_INTERVAL_SECONDS = 180

# Maximum runtime for the claude subprocess. Beyond this, we kill claude
# and block the task with a timeout reason so the dispatcher can re-route.
DEFAULT_CLAUDE_TIMEOUT_SECONDS = 1800

# Shared reference so the signal handler can terminate the child before
# the wrapper itself exits (prevents orphaned claude processes when the
# gateway sends SIGTERM for max-runtime / stale-claim cleanup).
_current_child: list[subprocess.Popen | None] = [None]


def _print(msg: str) -> None:
    """Stdout-only print so dispatcher's log capture sees it."""
    print(msg, flush=True)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _install_signal_forensics() -> None:
    """Trap fatal signals and dump diagnostic info before exiting.

    Goal: identify who is sending SIGTERM to this wrapper. Logs the
    elapsed time since start, our pid/ppid, parent's cmdline, and the
    open file descriptors. Then re-raises the signal so the default
    disposition (terminate) still applies.
    """
    t_start = time.time()
    pid = os.getpid()

    def _ppid_cmdline() -> str:
        try:
            with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
                return f.read().decode("utf-8", "replace").replace("\x00", " ").strip()
        except OSError:
            return "(unreadable)"

    def _handle(signum: int, _frame: Any) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = f"signal({signum})"
        elapsed = time.time() - t_start
        try:
            ppid = os.getppid()
        except OSError:
            ppid = -1
        _eprint(
            f"[claude-runner] !! received {sig_name} ({signum}) after "
            f"{elapsed:.1f}s pid={pid} ppid={ppid} "
            f"parent_cmdline={_ppid_cmdline()!r}"
        )
        # Best-effort child cleanup before we die — prevents orphaned
        # claude processes when the gateway kills us via enforce_max_runtime.
        proc = _current_child[0]
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        # Restore default handler so the re-raise actually kills us
        signal.signal(signum, signal.SIG_DFL)
        os.kill(pid, signum)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT):
        try:
            signal.signal(sig, _handle)
        except (OSError, ValueError):
            pass


def _run_hermes_cli(argv: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    """Run a ``hermes kanban ...`` CLI invocation. Resolves the ``hermes``
    binary the same way the dispatcher does (PATH first, then
    ``python -m hermes_cli.main``)."""
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        cmd = [hermes_bin, *argv]
    else:
        cmd = [sys.executable, "-m", "hermes_cli.main", *argv]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )


def _kanban_show(task_id: str) -> dict[str, Any]:
    res = _run_hermes_cli(["kanban", "show", task_id, "--json"])
    if res.returncode != 0:
        raise RuntimeError(
            f"`hermes kanban show {task_id} --json` exited {res.returncode}: "
            f"{res.stderr.strip()[:300]}"
        )
    try:
        payload = json.loads(res.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"kanban show returned non-JSON: {e}; head={res.stdout[:200]!r}")
    # ``hermes kanban show --json`` wraps the task fields under a ``task`` key
    # alongside ``events`` / ``comments`` / ``latest_summary``. Older callers
    # used to pre-unwrap; we do it here so ``_build_goal`` can index
    # ``title`` / ``body`` directly.
    if isinstance(payload, dict) and "task" in payload and isinstance(payload["task"], dict):
        return payload["task"]
    return payload


def _kanban_block(task_id: str, reason: str) -> None:
    # The CLI takes positional reason words; pass as a single string and
    # let argparse collapse it via nargs='*'.
    res = _run_hermes_cli(["kanban", "block", task_id, reason])
    if res.returncode != 0:
        _eprint(
            f"kanban block failed (rc={res.returncode}): {res.stderr.strip()[:300]}"
        )


def _kanban_complete(task_id: str, *, summary: str, metadata: dict[str, Any]) -> None:
    argv = ["kanban", "complete", task_id]
    if summary:
        # Truncate to a reasonable size — kanban_complete tolerates large
        # blobs but they bloat the board.
        argv.extend(["--summary", summary[:2000]])
        argv.extend(["--result", summary[:2000]])
    if metadata:
        argv.extend(["--metadata", json.dumps(metadata, ensure_ascii=False)])
    res = _run_hermes_cli(argv)
    if res.returncode != 0:
        _eprint(
            f"kanban complete failed (rc={res.returncode}): {res.stderr.strip()[:300]}"
        )


def _kanban_heartbeat(task_id: str, note: str | None = None) -> None:
    argv = ["kanban", "heartbeat", task_id]
    if note:
        argv.extend(["--note", note])
    # Best effort; never raise from the heartbeat thread.
    try:
        _run_hermes_cli(argv)
    except Exception as e:
        _eprint(f"heartbeat error (non-fatal): {e}")


class _HeartbeatThread(threading.Thread):
    """Daemon thread that pings ``hermes kanban heartbeat`` periodically.

    Exits when ``stop()`` is called or when the main thread terminates
    (daemon=True). Failures are swallowed so a transient kanban CLI
    glitch doesn't take the worker down — the claim TTL will eventually
    expire and the dispatcher will re-claim.
    """

    def __init__(self, task_id: str, interval: int = HEARTBEAT_INTERVAL_SECONDS):
        super().__init__(daemon=True, name=f"kanban-heartbeat-{task_id}")
        self._task_id = task_id
        self._interval = interval
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # Sleep first so we don't beat immediately on top of the
            # dispatcher's initial claim event.
            if self._stop_event.wait(timeout=self._interval):
                break
            _kanban_heartbeat(self._task_id, note="claude-runner heartbeat")

    def stop(self) -> None:
        self._stop_event.set()


def _build_goal(task: dict[str, Any], max_turns: int) -> str:
    """Compose the ``claude -p`` prompt from the task body.

    Default shape::

      /goal {task body}

      <stop after N turns>

    The ``/goal`` slash command is how claude-code itself frames an
    objective-driven loop in print mode — see
    ``skills/autonomous-ai-agents/claude-code/SKILL.md`` for the full
    pattern. Profile-level overrides can be added later by reading a
    template from the profile's SOUL.md.
    """
    title = task.get("title") or ""
    body = task.get("body") or task.get("description") or ""
    parts: list[str] = []
    if title:
        parts.append(title)
    if body and body.strip() != title.strip():
        parts.append(body)
    objective = "\n\n".join(p for p in parts if p).strip() or "(no task body provided)"
    return f"/goal {objective} or stop after {max_turns} turns"


def _resolve_claude_bin() -> str:
    bin_path = shutil.which("claude")
    if not bin_path:
        # Last-ditch defaults that match the box this typically runs on.
        for cand in ("/home/amsterdam/.local/bin/claude", "/usr/local/bin/claude"):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        raise RuntimeError(
            "`claude` CLI not found on PATH. Install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code) or extend the "
            "gateway service's PATH to include it."
        )
    return bin_path


def _load_runner_cfg(profile_name: str | None) -> dict[str, Any]:
    """Read the ``runner:`` block from the active profile's config.yaml.

    Falls back to an empty dict if the profile can't be located — the
    caller still has sensible per-key defaults baked into command
    assembly below.
    """
    if not profile_name:
        return {}
    try:
        from hermes_cli.profiles import _read_runner_config
        cfg = _read_runner_config(profile_name)
        # The wrapper only runs for kind == "claude-code"; defensively
        # accept any kind here and just use the parameter fields.
        return cfg or {}
    except Exception as e:
        _eprint(f"could not read runner config for profile {profile_name!r}: {e}")
        return {}


def _build_claude_argv(goal: str, cfg: dict[str, Any]) -> list[str]:
    claude_bin = _resolve_claude_bin()
    argv: list[str] = [
        claude_bin,
        "-p", goal,
        "--output-format", str(cfg.get("output_format", "json")),
        "--max-turns", str(int(cfg.get("max_turns", 30))),
    ]
    if cfg.get("dangerously_skip_permissions"):
        argv.append("--dangerously-skip-permissions")
    allowed_tools = cfg.get("allowed_tools")
    if allowed_tools:
        argv.extend(["--allowedTools", str(allowed_tools)])
    model = cfg.get("model")
    if model:
        argv.extend(["--model", str(model)])
    return argv


def main() -> int:
    _install_signal_forensics()
    try:
        task_id = os.environ["HERMES_KANBAN_TASK"]
    except KeyError:
        _eprint("HERMES_KANBAN_TASK env var is required")
        return 2
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", os.getcwd())
    profile = os.environ.get("HERMES_PROFILE")

    cfg = _load_runner_cfg(profile)

    # Fetch the task body so we can build the goal prompt.
    try:
        task = _kanban_show(task_id)
    except Exception as e:
        _eprint(f"could not fetch task {task_id}: {e}")
        _kanban_block(task_id, f"claude-runner: fetch task failed: {e}")
        return 1

    max_turns = int(cfg.get("max_turns", 30))
    goal = _build_goal(task, max_turns)

    try:
        argv = _build_claude_argv(goal, cfg)
    except RuntimeError as e:
        _kanban_block(task_id, f"claude-runner: {e}")
        return 1

    cwd = workspace if os.path.isdir(workspace) else None
    timeout = int(cfg.get("timeout", DEFAULT_CLAUDE_TIMEOUT_SECONDS))

    _print(f"[claude-runner] task={task_id} workspace={cwd}")
    _print(f"[claude-runner] argv={argv!r}")
    _print(
        f"[claude-runner] env keys: HOME={os.environ.get('HOME')!r} "
        f"HERMES_HOME={os.environ.get('HERMES_HOME')!r} "
        f"HERMES_PROFILE={os.environ.get('HERMES_PROFILE')!r} "
        f"PATH_head={os.environ.get('PATH', '').split(':')[0]!r}"
    )

    heartbeat = _HeartbeatThread(task_id)
    heartbeat.start()
    t_start = time.time()
    try:
        try:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            heartbeat.stop()
            _kanban_block(
                task_id,
                "claude-runner: claude binary disappeared between resolve and exec",
            )
            return 1
        try:
            child_pgid = os.getpgid(child.pid)
            child_sid = os.getsid(child.pid)
        except ProcessLookupError:
            child_pgid = child_sid = -1
        _print(
            f"[claude-runner] claude started pid={child.pid} pgid={child_pgid} "
            f"sid={child_sid}"
        )
        _current_child[0] = child
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate()
            _kanban_block(
                task_id,
                f"claude-runner: subprocess timed out after {timeout}s",
            )
            return 1
        finally:
            _current_child[0] = None
    finally:
        heartbeat.stop()

    elapsed = time.time() - t_start
    returncode = child.returncode
    _print(
        f"[claude-runner] claude finished elapsed={elapsed:.1f}s returncode={returncode} "
        f"stdout_len={len(stdout or '')} stderr_len={len(stderr or '')}"
    )
    if stdout:
        _print(f"[claude-runner] STDOUT tail: {stdout[-2000:]!r}")
    if stderr:
        _eprint(f"[claude-runner] STDERR tail: {stderr[-2000:]!r}")

    # Snapshot raw outputs to a side file for postmortem — survives the
    # log rotation cap on the kanban log file.
    try:
        snap_dir = os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "kanban_claude_runner_logs",
        )
        os.makedirs(snap_dir, exist_ok=True)
        with open(os.path.join(snap_dir, f"{task_id}.stdout"), "w") as f:
            f.write(stdout or "")
        with open(os.path.join(snap_dir, f"{task_id}.stderr"), "w") as f:
            f.write(stderr or "")
    except OSError as e:
        _eprint(f"[claude-runner] could not snapshot outputs: {e}")

    if returncode != 0:
        _kanban_block(
            task_id,
            f"claude exit {returncode}: "
            f"{(stderr or '').strip()[:300] or (stdout or '').strip()[:300]}",
        )
        return 1

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        _kanban_block(
            task_id,
            f"claude JSON parse failed ({e}); head={stdout[:200]!r}",
        )
        return 1

    if not isinstance(data, dict):
        _kanban_block(task_id, f"claude JSON not an object: {type(data).__name__}")
        return 1

    subtype = data.get("subtype")
    result_text = data.get("result") or ""
    metadata = {
        "runner": "claude-code",
        "session_id": data.get("session_id"),
        "num_turns": data.get("num_turns"),
        "total_cost_usd": data.get("total_cost_usd"),
        "duration_ms": data.get("duration_ms"),
        "duration_api_ms": data.get("duration_api_ms"),
        "stop_reason": data.get("stop_reason"),
        "terminal_reason": data.get("terminal_reason"),
        "model_usage": data.get("modelUsage"),
    }
    # Strip None values so the metadata JSON stays tidy in the board UI.
    metadata = {k: v for k, v in metadata.items() if v is not None}

    if subtype == "success" and not data.get("is_error"):
        _kanban_complete(task_id, summary=result_text, metadata=metadata)
        return 0

    api_err = data.get("api_error_status")
    reason = (
        f"claude {subtype or 'unknown'}: "
        f"{(result_text or '')[:300]}"
        + (f" (api_error_status={api_err})" if api_err else "")
    )
    metadata_block_note = {
        "outcome_metadata": metadata,
    }
    # Block with metadata embedded in the reason text — kanban block
    # CLI doesn't accept --metadata directly. The full structured info
    # is still in the stdout-captured log file as a fallback.
    _kanban_block(task_id, f"{reason} :: {json.dumps(metadata_block_note, ensure_ascii=False)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
