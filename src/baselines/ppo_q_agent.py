"""
Canonical re-export of PPOQAgent from src.agents.ppo_q_agent.
Preserves backward compatibility across baseline runners.
"""
from src.agents.ppo_q_agent import PPOQAgent

__all__ = ["PPOQAgent"]
