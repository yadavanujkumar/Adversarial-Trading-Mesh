"""
src/agents/__init__.py — exposes the three specialist agents.
"""
from .bull_agent import run_bull_agent
from .bear_agent import run_bear_agent
from .judge_agent import run_judge_agent

__all__ = ["run_bull_agent", "run_bear_agent", "run_judge_agent"]
