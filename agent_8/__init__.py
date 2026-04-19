"""
agent_8/__init__.py
====================
Expose l'interface publique d'Agent 8 - Quality Gate
pour une intégration facile dans un pipeline multi-agents.

Usage :
    from agent_8 import run_agent8
    result_state = run_agent8(state)
"""

from .main import run_agent8

__all__ = ["run_agent8"]
__version__ = "1.0.0"
__author__ = "NLP Dataset Intelligence Engine"
