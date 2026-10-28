import numpy as np
from typing import Dict, Any, Tuple
from src.environment.gridworld import GridWorld
from src.baselines.classical_agent import ClassicalAgent
from src.quantum.quantum_agent import QuantumAgent
from src.evaluation.metrics import ExperimentMetrics, EpisodeResult, HypothesisTester

class BenchmarkRunner:
    """
    Executes controlled paired experiments comparing Classical Model-Based Agent vs Quantum-Assisted Agent.
    Supports ablation configurations for the Quantum Agent.
    """
    def __init__(
        self,
        grid_width: int = 5,
        grid_height: int = 5,
        obstacles: list = None,
        traps: list = None,
        horizon: int = 3,
        episodes: int = 30,
        eval_budget: int = 8,
        shots: int = 128,
        seed: int = 42,
        enable_oracle: bool = True,
        enable_interference: bool = True
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles or [(1, 1), (1, 2), (2, 3), (3, 1)]
        self.traps = traps or [(2, 1)]
        self.horizon = horizon
        self.episodes = episodes
        self.eval_budget = eval_budget
        self.shots = shots
        self.seed = seed
        self.enable_oracle = enable_oracle
        self.enable_interference = enable_interference

    def create_env(self) -> GridWorld:
        return GridWorld(
            width=self.grid_width,
            height=self.grid_height,
            start_pos=(0, 0),
            goal_pos=(self.grid_height - 1, self.grid_width - 1),
            obstacles=self.obstacles,
            traps=self.traps,
            step_penalty=-0.02,
            goal_reward=1.0,
            obstacle_penalty=-0.2,
            trap_penalty=-0.8,
            max_steps=35
        )

    def run_agent_experiment(self, agent_type: str, seed: int) -> ExperimentMetrics:
        np.random.seed(seed)
        env = self.create_env()

        if agent_type == "classical":
            agent = ClassicalAgent(
                env=env, horizon=self.horizon, eval_budget=self.eval_budget
            )
            metrics = ExperimentMetrics(agent_name="Classical World Model Agent")
        else:
            agent = QuantumAgent(
                env=env,
                horizon=self.horizon,
                shots=self.shots,
                enable_oracle=self.enable_oracle,
                enable_interference=self.enable_interference
            )
            metrics = ExperimentMetrics(
                agent_name="Quantum World Model Agent",
                qubit_count=agent.world_model.num_qubits,
                circuit_depth=agent.world_model.last_circuit_depth
            )

        for ep in range(self.episodes):
            pos = env.reset()
            total_reward = 0.0
            done = False
            evals_before = (
                agent.world_model.total_evaluations
                if agent_type == "classical"
                else agent.world_model.total_circuit_evaluations
            )

            while not done:
                action = agent.select_action(pos)
                next_pos, reward, done, info = env.step(action)
                agent.update(pos, action, reward, next_pos, done)
                total_reward += reward
                pos = next_pos

            evals_after = (
                agent.world_model.total_evaluations
                if agent_type == "classical"
                else agent.world_model.total_circuit_evaluations
            )

            result = EpisodeResult(
                episode_index=ep,
                total_reward=total_reward,
                steps=env.steps_taken,
                success=info.get("reached_goal", False),
                evaluations_count=evals_after - evals_before,
                shots_count=self.shots if agent_type == "quantum" else 0
            )
            metrics.episode_results.append(result)

        return metrics

    def run_benchmark(self) -> Dict[str, Any]:
        classical_metrics = self.run_agent_experiment("classical", seed=self.seed)
        quantum_metrics = self.run_agent_experiment("quantum", seed=self.seed)

        stats_report = HypothesisTester.compare_agents(classical_metrics, quantum_metrics)

        return {
            "classical_metrics": classical_metrics,
            "quantum_metrics": quantum_metrics,
            "hypothesis_report": stats_report
        }
