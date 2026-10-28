import pytest
from src.environment.gridworld import GridWorld, UP, RIGHT, DOWN, LEFT

def test_gridworld_initialization():
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4))
    assert env.width == 5
    assert env.height == 5
    assert env.num_states == 25
    assert env.reset() == (0, 0)

def test_gridworld_step_and_boundary():
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4), step_penalty=-0.01)
    
    # Try moving UP from (0,0) -> boundary collision
    pos, reward, done, info = env.step(UP)
    assert pos == (0, 0)
    assert reward == env.obstacle_penalty

    # Move RIGHT to (0,1)
    pos, reward, done, info = env.step(RIGHT)
    assert pos == (0, 1)
    assert reward == -0.01
    assert not done

def test_gridworld_simulate_step():
    env = GridWorld(width=5, height=5, start_pos=(0, 0), goal_pos=(4, 4))
    next_pos, reward, done = env.simulate_step((0, 0), DOWN)
    assert next_pos == (1, 0)
    assert reward == env.step_penalty
    assert not done
