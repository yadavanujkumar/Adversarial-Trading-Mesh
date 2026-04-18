"""
src/memory.py — persistent agent memory backed by a JSON file.

Each agent accumulates "lessons" from post-mortem analyses.  These lessons
are injected into the agent's system prompt at the start of each reasoning
cycle, allowing the swarm to genuinely learn from past failures without
relying on an external vector store.

Enterprise-grade additions
--------------------------
* asyncio.Lock prevents concurrent coroutines from corrupting the JSON file
  when multiple agents write lessons simultaneously after a stop-loss event.
* In-memory cache avoids disk I/O on every reasoning cycle read.
* Lessons are capped at settings.memory_max_lessons per agent to bound file growth.
* Atomic write via a temp file + os.replace() to avoid partial-write corruption.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency at module load time
def _get_settings() -> Any:
    from src.config import settings  # noqa: PLC0415
    return settings


# ── Internal state ─────────────────────────────────────────────────────────────

_lock = asyncio.Lock()                 # guards all file I/O
_cache: dict[str, Any] | None = None  # in-memory mirror of the JSON file


def _memory_path() -> Path:
    return _get_settings().memory_path_obj


def _max_lessons() -> int:
    return _get_settings().memory_max_lessons


# ── Synchronous I/O helpers (called while holding _lock) ──────────────────────

def _load_sync() -> dict[str, Any]:
    global _cache
    if _cache is not None:
        return _cache
    path = _memory_path()
    if path.exists():
        try:
            _cache = json.loads(path.read_text(encoding="utf-8"))
            return _cache
        except json.JSONDecodeError:
            logger.warning("Corrupt memory file at %s — starting fresh.", path)
    _cache = {"bull": [], "bear": [], "judge": []}
    return _cache


def _save_sync(data: dict[str, Any]) -> None:
    """Atomically write *data* to disk via temp-file + os.replace()."""
    global _cache
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".agent_memory_", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        _cache = data  # update in-memory mirror
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── Public API ─────────────────────────────────────────────────────────────────

def get_lessons(agent: str) -> list[dict[str, Any]]:
    """Return the stored lessons for *agent* (most recent N only, sync)."""
    data = _load_sync()
    return data.get(agent, [])[-_max_lessons():]


async def async_get_lessons(agent: str) -> list[dict[str, Any]]:
    """Async-safe variant — preferred inside coroutines."""
    async with _lock:
        data = _load_sync()
    return data.get(agent, [])[-_max_lessons():]


def append_lesson(agent: str, lesson: dict[str, Any]) -> None:
    """
    Append a lesson entry for *agent* (sync wrapper retained for compatibility).

    For concurrent-safe usage inside coroutines, call async_append_lesson().
    """
    data = _load_sync()
    data.setdefault(agent, [])
    lesson_with_ts = {**lesson, "timestamp": datetime.now(timezone.utc).isoformat()}
    data[agent].append(lesson_with_ts)
    # Trim to max lessons
    data[agent] = data[agent][-_max_lessons():]
    _save_sync(data)
    logger.info("Memory updated for agent=%s: %s", agent, lesson.get("category", "unknown"))


async def async_append_lesson(agent: str, lesson: dict[str, Any]) -> None:
    """
    Append a lesson entry for *agent* — async-safe, holds the I/O lock.

    Preferred over append_lesson() when called from a coroutine.
    """
    async with _lock:
        data = _load_sync()
        data.setdefault(agent, [])
        lesson_with_ts = {**lesson, "timestamp": datetime.now(timezone.utc).isoformat()}
        data[agent].append(lesson_with_ts)
        data[agent] = data[agent][-_max_lessons():]
        _save_sync(data)
    logger.info("Memory updated for agent=%s: %s", agent, lesson.get("category", "unknown"))


def format_lessons_for_prompt(agent: str) -> str:
    """
    Render the stored lessons as a compact bullet list for injection into
    a system prompt.  Returns an empty string when no lessons exist yet.
    """
    lessons = get_lessons(agent)
    if not lessons:
        return ""
    lines = ["## Past Failure Lessons (most recent first)"]
    for entry in reversed(lessons):
        ts = entry.get("timestamp", "?")[:10]
        cat = entry.get("category", "other")
        summary = entry.get("summary", "—")
        lines.append(f"- [{ts}] ({cat}): {summary}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Invalidate the in-memory cache (useful for testing)."""
    global _cache
    _cache = None
