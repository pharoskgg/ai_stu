"""5×5 迷宫中的 Q-Learning 实验。

运行：
    python 01_q_learning_maze.py

程序会训练智能体、打印学习到的最优策略，并在当前目录生成
``q_learning_result.png``（需要安装 matplotlib）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# 动作编码：上、右、下、左
ACTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)
ACTION_SYMBOLS = np.array(["↑", "→", "↓", "←"])


@dataclass
class MazeEnv:
    """确定性的 5×5 网格迷宫环境。"""

    rows: int = 5
    cols: int = 5

    def __post_init__(self) -> None:
        self.start = (0, 0)
        self.goal = (4, 4)
        # X 为障碍物；布局保留了一条从左上到右下的可行路径。
        self.walls = {(0, 2), (1, 0), (1, 2), (1, 4), (3, 0), (3, 1), (3, 3)}
        self.state = self.start

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @staticmethod
    def _state_id(position: tuple[int, int]) -> int:
        return position[0] * 5 + position[1]

    def reset(self) -> int:
        self.state = self.start
        return self._state_id(self.state)

    def step(self, action: int) -> tuple[int, float, bool]:
        """执行动作，返回 (next_state, reward, done)。"""
        dr, dc = ACTIONS[action]
        row, col = int(self.state[0] + dr), int(self.state[1] + dc)
        candidate = (row, col)

        # 撞墙或出界：留在原地，并给更低的惩罚。
        if not (0 <= row < self.rows and 0 <= col < self.cols) or candidate in self.walls:
            return self._state_id(self.state), -10.0, False

        self.state = candidate
        if self.state == self.goal:
            return self._state_id(self.state), 100.0, True
        return self._state_id(self.state), -1.0, False


def train_q_learning(
    env: MazeEnv,
    episodes: int = 1200,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    max_steps: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 ε-greedy 策略训练，返回 Q 表、每回合回报和步数。"""
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.n_states, len(ACTIONS)), dtype=float)
    rewards = np.zeros(episodes, dtype=float)
    steps_per_episode = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        # 线性降低探索率：前期充分探索，后期稳定利用已学策略。
        ratio = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + ratio * (epsilon_end - epsilon_start)
        state = env.reset()

        for step in range(1, max_steps + 1):
            if rng.random() < epsilon:
                action = int(rng.integers(len(ACTIONS)))
            else:
                # 并列最大值随机选择，避免固定的动作偏置。
                best_actions = np.flatnonzero(q_table[state] == q_table[state].max())
                action = int(rng.choice(best_actions))

            next_state, reward, done = env.step(action)
            target = reward if done else reward + gamma * q_table[next_state].max()
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state
            rewards[episode] += reward

            if done:
                steps_per_episode[episode] = step
                break
        else:
            steps_per_episode[episode] = max_steps

    return q_table, rewards, steps_per_episode


def greedy_path(env: MazeEnv, q_table: np.ndarray, max_steps: int = 30) -> list[tuple[int, int]]:
    """按 Q 表的贪心策略走出一条路径，并防止意外死循环。"""
    env.reset()
    path = [env.state]
    visited = {env.state}

    for _ in range(max_steps):
        state = env._state_id(env.state)
        action = int(np.argmax(q_table[state]))
        _, _, done = env.step(action)
        path.append(env.state)
        if done:
            return path
        if env.state in visited:
            raise RuntimeError("贪心策略形成循环；请增加训练回合数后重试。")
        visited.add(env.state)

    raise RuntimeError("未能在最大步数内到达终点。")


def print_policy(env: MazeEnv, q_table: np.ndarray, path: list[tuple[int, int]]) -> None:
    """以字符形式打印迷宫与学得策略。"""
    path_set = set(path)
    print("\n学得的贪心策略（S: 起点，G: 终点，X: 障碍物，·: 最优路径）")
    for row in range(env.rows):
        symbols: list[str] = []
        for col in range(env.cols):
            position = (row, col)
            if position == env.start:
                symbols.append(" S ")
            elif position == env.goal:
                symbols.append(" G ")
            elif position in env.walls:
                symbols.append(" X ")
            elif position in path_set:
                symbols.append(" · ")
            else:
                state = env._state_id(position)
                symbols.append(f" {ACTION_SYMBOLS[np.argmax(q_table[state])]} ")
        print("".join(symbols))
    print("\n最优路径：" + " -> ".join(map(str, path)))


def save_figure(
    env: MazeEnv, rewards: np.ndarray, steps: np.ndarray, path: list[tuple[int, int]], output: Path
) -> None:
    """保存训练曲线和最终路径图。未安装 matplotlib 时跳过。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("未安装 matplotlib，已跳过图片生成。可执行 pip install matplotlib 后重试。")
        return

    fig, (ax_curve, ax_maze) = plt.subplots(1, 2, figsize=(13, 5))
    window = min(50, len(rewards))
    average_reward = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax_curve.plot(rewards, alpha=0.25, label="Episode reward")
    ax_curve.plot(np.arange(window - 1, len(rewards)), average_reward, linewidth=2, label=f"{window}-episode moving average")
    ax_curve.set(title="Q-Learning Training Return", xlabel="Episode", ylabel="Total reward")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend()

    grid = np.zeros((env.rows, env.cols))
    for row, col in env.walls:
        grid[row, col] = 1
    ax_maze.imshow(grid, cmap=ListedColormap(["#f8fafc", "#334155"]), vmin=0, vmax=1)
    ax_maze.set(title=f"Greedy path ({len(path) - 1} steps)", xticks=range(env.cols), yticks=range(env.rows))
    ax_maze.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax_maze.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax_maze.grid(which="minor", color="#94a3b8", linewidth=1)
    ax_maze.tick_params(which="minor", bottom=False, left=False)
    path_rows, path_cols = zip(*path)
    ax_maze.plot(path_cols, path_rows, "o-", color="#2563eb", linewidth=2.5, markersize=6, label="Path")
    ax_maze.scatter(*env.start[::-1], color="#16a34a", s=120, zorder=3, label="Start S")
    ax_maze.scatter(*env.goal[::-1], color="#dc2626", s=120, zorder=3, label="Goal G")
    ax_maze.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"训练图已保存：{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5×5 迷宫 Q-Learning 实验")
    parser.add_argument("--episodes", type=int, default=1200, help="训练回合数（默认：1200）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    args = parser.parse_args()

    env = MazeEnv()
    q_table, rewards, steps = train_q_learning(env, episodes=args.episodes, seed=args.seed)
    path = greedy_path(env, q_table)
    print(f"训练完成：{args.episodes} 回合")
    print(f"最后 100 回合平均回报：{rewards[-100:].mean():.2f}")
    print(f"最后 100 回合平均步数：{steps[-100:].mean():.2f}")
    print_policy(env, q_table, path)
    save_figure(env, rewards, steps, path, Path(__file__).with_name("q_learning_result.png"))


if __name__ == "__main__":
    main()
