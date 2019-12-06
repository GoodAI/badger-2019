import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def generate_random_runs(num_runs: int, num_steps: int, amplitude: float, noise_amp: float) -> np.ndarray:
    """
    Args:
        num_runs:
        num_steps:
        amplitude:
        noise_amp:

    Returns: [num_runs, num_steps] array of results
    """
    time = np.arange(num_steps)
    res = np.log(time + 1) * amplitude

    result = np.zeros((num_runs, num_steps))
    for run in range(num_runs):
        run_noise_amp = np.random.randn() + noise_amp
        result[run] = res + np.random.randn(result[run].size) * run_noise_amp

    return result


def preprocess(data: np.ndarray, smoothing_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # minn = smooth(data.min(axis=0), smoothing_size)[:-smoothing_size//2]
    # maxx = smooth(data.max(axis=0), smoothing_size)[:-smoothing_size//2]
    minn = data.min(axis=0)[:-smoothing_size//2]
    maxx = data.max(axis=0)[:-smoothing_size//2]
    meann = smooth(data.mean(axis=0), smoothing_size)[:-smoothing_size//2]

    return minn, maxx, meann, np.arange(data.shape[1])[:-smoothing_size//2]


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def add_data(data: np.ndarray, label: str, color: str, smoothing_size: int = 100):
    minn, maxx, meann, time = preprocess(data, smoothing_size)

    plt.fill_between(time, minn, maxx, color=color, alpha=0.1)
    plt.plot(time, meann, linewidth=0.5, color=color, label=label)
    plt.legend()


def run_loader_rewards(filenames: List[str]):
    rewards = []
    for fname in filenames:
        with open(f"data/stats/{fname}.rewards", 'r') as f:
            r = f.readline().split(",")[:-1]
            rewards.append([float(k) for k in r])

    return np.array(rewards)


def main():
    files_ddpg = ["Run_19-12-06_1515"]
    files_global = ["Run_19-12-06_1531"]
    files_atoc = ["Run_19-12-06_1530"]

    rewards_ddpg = run_loader_rewards(files_ddpg)
    rewards_global = run_loader_rewards(files_global)
    rewards_atoc = run_loader_rewards(files_atoc)

    plt.figure()
    add_data(rewards_ddpg, 'DDPG independent', 'red')
    add_data(rewards_global, 'DDPG global', 'green')
    add_data(rewards_atoc, 'ATOC', 'blue')

    plt.title('Convergence of learning on 4 agents & 4 landmarks')

    plt.xlabel('Simulation steps')
    plt.ylabel('Mean reward')

    if not os.path.exists('data/figures/'):
        os.makedirs('data/figures/')
    plt.savefig('data/figures/4_landmarks.png', format='png')
    plt.show()


if __name__ == '__main__':

    main()

    # num_steps = 1000000
    # num_runs = 10
    #
    # ddpg_independent = generate_random_runs(num_runs, num_steps, 12, 2)
    # ddpg_global = generate_random_runs(num_runs, num_steps, 8, 3)
    # atoc = generate_random_runs(num_runs, num_steps, 14, 1.5)
    #
    # plt.figure()
    # add_data(ddpg_independent, 'DDPG independent', 'red')
    # add_data(ddpg_global, 'DDPG global', 'green')
    # add_data(atoc, 'ATOC', 'blue')
    #
    # plt.title('Convergence of learning on 4 agents & 4 landmarks')
    #
    # plt.xlabel('Simulation steps')
    # plt.ylabel('Mean reward')
    # plt.show()



