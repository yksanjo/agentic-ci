"""
Agentic CI: Intelligent CI System for the Agentic Era

Idea inspired by Peter Steinberger's vision for intelligent CI systems.
CI that understands context, not just commands.
"""

__version__ = "0.1.0"
__author__ = "yksanjo"
__credits__ = ["Peter Steinberger (original concept)"]

from .core.analyzer import ChangeAnalyzer
from .core.llm_client import LLMClient

__all__ = ["ChangeAnalyzer", "LLMClient"]
