"""
src/memory.py — persistent agent memory backed by a JSON file.

Each agent accumulates "lessons" from post-mortem analyses.  These lessons
are injected into the agent's system prompt at the start of each reasoning
cycle, allowing the swarm to genuinely learn from past failures without
relying on an external vector store.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_PATH = Path(os.getenv("MEMORY_PATH", "agent_memory.json"))


def _load() -> dict[str, Any]:
    if _MEMORY_PATH.exists():
        try:
            return json.loads(_MEMORY_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt memory file — starting fresh.")
    return {"bull": [], "bear": [], "judge": []}


def _save(data: dict[str, Any]) -> None:
    _MEMORY_PATH.write_text(json.dumps(data, indent=2))


def get_lessons(agent: str) -> list[dict[str, Any]]:
    """Return the stored lessons for *agent* (most recent 10 only)."""
    return _load().get(agent, [])[-10:]


def append_lesson(agent: str, lesson: dict[str, Any]) -> None:
    """
    Append a lesson entry for *agent*.

    Parameters
    ----------
    agent:  "bull" | "bear" | "judge"
    lesson: arbitrary dict; typically contains 'timestamp', 'category',
            'summary', and optionally 'risk_weight_delta'.
    """
    data = _load()
    data.setdefault(agent, [])
    lesson_with_ts = {**lesson, "timestamp": datetime.now(timezone.utc).isoformat()}
    data[agent].append(lesson_with_ts)
    _save(data)
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
