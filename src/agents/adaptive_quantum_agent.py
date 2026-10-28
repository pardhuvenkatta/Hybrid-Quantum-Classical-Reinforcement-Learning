"""
Re-export shim for AdaptiveQuantumAgent from canonical location src.quantum.adaptive_quantum_agent.
Preserves backward compatibility across baseline runners.
"""
from src.quantum.adaptive_quantum_agent import *  # noqa: F401, F403
from src.quantum.adaptive_quantum_agent import AdaptiveQuantumAgent

__all__ = ["AdaptiveQuantumAgent"]
