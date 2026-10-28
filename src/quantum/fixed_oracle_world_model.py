"""
Re-export FixedQuantumWorldModel from canonical location src.quantum.fixed_oracle.
Preserves backward compatibility across baseline runners and tests.
"""
from src.quantum.fixed_oracle import FixedQuantumWorldModel

__all__ = ["FixedQuantumWorldModel"]
