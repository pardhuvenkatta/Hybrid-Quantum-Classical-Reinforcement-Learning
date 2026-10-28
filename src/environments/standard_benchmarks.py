import numpy as np
from typing import Tuple, List, Dict, Any, Optional

ACTIONS = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT for Grid/Maze environments

class BaseBenchmarkEnv:
    """Base interface for all benchmark environments."""
    def __init__(self, name: str):
        self.name = name
        self.num_actions = 4
        self.num_states = 16
        self.width = 4
        self.height = 4
        self.goal_pos = (3, 3)
        self.start_pos = (0, 0)
        self.obstacles = set()
        self.traps = set()
        self.step_penalty = -0.05
        self.revisit_penalty = 0.0
        self.obstacle_penalty = -0.30
        self.trap_penalty = -0.80
        self.goal_reward = 1.00

    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def simulate_step(self, state: Any, action: int) -> Tuple[Any, float, bool]:
        raise NotImplementedError

    def pos_to_state(self, pos: Any) -> int:
        return 0


class GridWorldMaze(BaseBenchmarkEnv):
    """Grid World / Maze Navigation benchmark for variable dimensions."""
    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        traps: Optional[List[Tuple[int, int]]] = None,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
        step_penalty: float = -0.05,
        revisit_penalty: float = -0.15,
        obstacle_penalty: float = -0.30,
        trap_penalty: float = -0.80,
        goal_reward: float = 1.00,
        max_steps: int = 100
    ):
        super().__init__(f"Maze-{width}x{height}")
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else (height - 1, width - 1)

        self.obstacles = set(obstacles) if obstacles is not None else set()
        self.traps = set(traps) if traps is not None else set()

        # If obstacles/traps not specified for larger mazes, generate deterministic patterns
        if obstacles is None and width >= 4:
            self._generate_default_layout()

        self.step_penalty = step_penalty
        self.revisit_penalty = revisit_penalty
        self.obstacle_penalty = obstacle_penalty
        self.trap_penalty = trap_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.agent_pos = self.start_pos
        self.visited_cells = set([self.start_pos])
        self.current_step = 0

    def _generate_default_layout(self):
        w, h = self.width, self.height
        obs = set()
        trps = set()
        # Create deterministic obstacles along internal walls
        for r in range(1, h - 1, 2):
            for c in range(1, w - 1, 2):
                if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                    obs.add((r, c))
        # Create deterministic traps near corners
        if h > 4 and w > 4:
            trps.add((2, 2))
            trps.add((h - 2, 1))
        self.obstacles = obs
        self.traps = trps

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.width + c

    def state_to_pos(self, state_idx: int) -> Tuple[int, int]:
        r = state_idx // self.width
        c = state_idx % self.width
        return (r, c)

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos
        self.visited_cells = set([self.start_pos])
        self.current_step = 0
        return self.agent_pos

    def _compute_next_pos(self, current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        r, c = current_pos
        if action == 0:    # UP
            r = max(0, r - 1)
        elif action == 1:  # RIGHT
            c = min(self.width - 1, c + 1)
        elif action == 2:  # DOWN
            r = min(self.height - 1, r + 1)
        elif action == 3:  # LEFT
            c = max(0, c - 1)

        candidate = (r, c)
        if candidate in self.obstacles:
            return current_pos  # Hit wall/obstacle -> stay in place
        return candidate

    def simulate_step(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        next_pos = self._compute_next_pos(current_pos, action)
        done = False
        reward = self.step_penalty

        if next_pos == self.goal_pos:
            reward += self.goal_reward
            done = True
        elif next_pos in self.traps:
            reward += self.trap_penalty
        elif next_pos == current_pos:
            reward += self.obstacle_penalty

        return next_pos, reward, done

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.current_step += 1
        next_pos, reward, done = self.simulate_step(self.agent_pos, action)

        if next_pos in self.visited_cells and not done and next_pos != self.agent_pos:
            reward += self.revisit_penalty

        self.visited_cells.add(next_pos)
        self.agent_pos = next_pos

        if self.current_step >= self.max_steps:
            done = True

        info = {
            "reached_goal": (self.agent_pos == self.goal_pos),
            "current_step": self.current_step,
            "visited_count": len(self.visited_cells)
        }
        return self.agent_pos, reward, done, info


class FrozenLakeEnv(GridWorldMaze):
    """Stochastic FrozenLake-v1 benchmark with slippery transition dynamics."""
    def __init__(self, is_slippery: bool = True, size: int = 4):
        # 4x4 or 8x8 layout
        obstacles = [(1, 1), (1, 3), (2, 3), (3, 0)] if size == 4 else None
        traps = [(1, 1)] if size == 4 else None
        super().__init__(
            width=size,
            height=size,
            obstacles=obstacles,
            traps=traps,
            start_pos=(0, 0),
            goal_pos=(size - 1, size - 1),
            step_penalty=-0.01,
            revisit_penalty=0.0,
            obstacle_penalty=-0.05,
            trap_penalty=-0.50,
            goal_reward=1.00,
            max_steps=100
        )
        self.name = f"FrozenLake-{size}x{size}-{'Slippery' if is_slippery else 'Deterministic'}"
        self.is_slippery = is_slippery

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        actual_action = action
        if self.is_slippery:
            # Slippery dynamics: 1/3 intended action, 1/3 perpendicular left, 1/3 perpendicular right
            r_val = np.random.rand()
            if r_val < 0.333:
                actual_action = (action - 1) % 4
            elif r_val < 0.666:
                actual_action = (action + 1) % 4

        return super().step(actual_action)


class CartPoleEnv(BaseBenchmarkEnv):
    """Discretized CartPole-v1 pole balancing control environment (discrete 2 actions)."""
    def __init__(self, max_steps: int = 500):
        super().__init__("Discretized CartPole-v1")
        self.num_actions = 2
        self.num_states = 16  # Discretized bin representation
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.max_steps = max_steps

        # Thresholds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360

        self.state = None
        self.steps = 0

    def reset(self) -> Tuple[int, int]:
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self._discretize_state(self.state)

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        # Discretize (x, x_dot, theta, theta_dot) into grid cell
        x, _, theta, _ = state
        r = 0 if theta < -0.05 else (1 if theta < 0.0 else (2 if theta < 0.05 else 3))
        c = 0 if x < -0.5 else (1 if x < 0.0 else (2 if x < 0.5 else 3))
        return (r, c)

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * 4 + c

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )

        reward = 1.0 if not done else 0.0
        disc_pos = self._discretize_state(self.state)
        return disc_pos, reward, done, {"reached_goal": self.steps >= 200}

    def simulate_step(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        # Approximate discrete step simulation for tree search
        r, c = current_pos
        dr = -1 if action == 0 else 1
        dc = -1 if action == 0 else 1
        nr = max(0, min(3, r + dr))
        nc = max(0, min(3, c + dc))
        done = (nr == 0 or nr == 3)
        reward = 1.0 if not done else 0.0
        return (nr, nc), reward, done


class AcrobotEnv(BaseBenchmarkEnv):
    """Discretized Acrobot-v1 double pendulum control environment (discrete 3 actions)."""
    def __init__(self, max_steps: int = 500):
        super().__init__("Discretized Acrobot-v1")
        self.num_actions = 3
        self.num_states = 16
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

    def reset(self) -> Tuple[int, int]:
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))
        self.steps = 0
        return (0, 0)

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * 4 + c

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        torque = action - 1.0  # -1, 0, +1
        theta1, theta2, dtheta1, dtheta2 = self.state

        dtheta1 += 0.05 * torque
        dtheta2 += 0.05 * torque
        theta1 += 0.05 * dtheta1
        theta2 += 0.05 * dtheta2

        self.state = np.array([theta1, theta2, dtheta1, dtheta2])
        height = -np.cos(theta1) - np.cos(theta2 + theta1)
        done = bool(height > 1.0 or self.steps >= self.max_steps)
        reward = 0.0 if height > 1.0 else -1.0

        r = min(3, max(0, int((theta1 + np.pi) / (2 * np.pi) * 4)))
        c = min(3, max(0, int((theta2 + np.pi) / (2 * np.pi) * 4)))
        return (r, c), reward, done, {"reached_goal": height > 1.0}

    def simulate_step(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        r, c = current_pos
        nr = max(0, min(3, r + (action - 1)))
        nc = max(0, min(3, c + (action - 1)))
        done = (nr == 3)
        reward = 0.0 if done else -1.0
        return (nr, nc), reward, done


class MountainCarEnv(BaseBenchmarkEnv):
    """MountainCar-v0 underpowered car hill climb (discrete 3 actions)."""
    def __init__(self, max_steps: int = 200):
        super().__init__("MountainCar-v0")
        self.num_actions = 3
        self.num_states = 16
        self.max_steps = max_steps
        self.position = -0.5
        self.velocity = 0.0
        self.steps = 0

    def reset(self) -> Tuple[int, int]:
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0.0
        self.steps = 0
        return self._discretize()

    def _discretize(self) -> Tuple[int, int]:
        r = int((self.position + 1.2) / 1.8 * 4)
        c = int((self.velocity + 0.07) / 0.14 * 4)
        return (min(3, max(0, r)), min(3, max(0, c)))

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * 4 + c

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        self.steps += 1
        force = action - 1  # -1, 0, 1
        self.velocity += 0.001 * force - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, -0.07, 0.07)
        self.position += self.velocity
        self.position = np.clip(self.position, -1.2, 0.6)
        if self.position == -1.2 and self.velocity < 0:
            self.velocity = 0.0

        done = bool(self.position >= 0.5 or self.steps >= self.max_steps)
        reward = 0.0 if self.position >= 0.5 else -1.0
        return self._discretize(), reward, done, {"reached_goal": self.position >= 0.5}

    def simulate_step(self, current_pos: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        r, c = current_pos
        dr = action - 1
        nr = max(0, min(3, r + dr))
        done = (nr == 3)
        reward = 0.0 if done else -1.0
        return (nr, c), reward, done


def create_benchmark_env(name: str, **kwargs) -> BaseBenchmarkEnv:
    """Factory helper to instantiate any benchmark environment."""
    name_lower = name.lower()
    if "cartpole" in name_lower:
        return CartPoleEnv(**kwargs)
    elif "acrobot" in name_lower:
        return AcrobotEnv(**kwargs)
    elif "mountaincar" in name_lower:
        return MountainCarEnv(**kwargs)
    elif "frozenlake" in name_lower:
        return FrozenLakeEnv(**kwargs)
    elif "maze" in name_lower or "grid" in name_lower:
        w = kwargs.pop("width", 6)
        h = kwargs.pop("height", 6)
        return GridWorldMaze(width=w, height=h, **kwargs)
    else:
        raise ValueError(f"Unknown environment benchmark name: {name}")
