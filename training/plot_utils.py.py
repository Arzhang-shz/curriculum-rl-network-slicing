import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute a simple moving average over 'data' with the given window size.
    Returns an array of length len(data) - window + 1.
    """
    if len(data) < window:
        return data  # Not enough data to average; return raw
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def plot_metric(
    pickle_path: str,
    ylabel: str,
    save_path: str,
    window: int = 100
):
    """
    Load a 1D array from 'pickle_path', compute its moving average, and plot it.
    Args:
      pickle_path: Path to a .txt or .pkl file containing a pickled list/array.
      ylabel: Label to use on the Y-axis.
      save_path: Where to save the resulting plot (.png or .eps).
      window: Window size for moving average.
    """
    if not os.path.isfile(pickle_path):
        print(f"Warning: {pickle_path} does not exist. Skipping.")
        return

    # Load data
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    data = np.array(data, dtype=float)
    avg_data = moving_average(data, window=window)
    x = np.arange(len(avg_data)) * window

    plt.figure()
    plt.plot(x, avg_data, label=ylabel)
    plt.xlabel("Episodes (×100)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Moving Avg, window={window})")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    # Modify these paths if your logs are in a different location
    base_log_dir = "results/logs"
    base_fig_dir = "results/figures"

    metrics = [
        ("R.txt", "Average Reward", "avg_reward.png"),
        ("S.txt", "Acceptance Rate", "avg_acceptance_rate.png"),
        ("Load_cpu_ram.txt", "CPU+RAM Load", "avg_cpu_ram_load.png"),
        ("Load_BW.txt", "Bandwidth Load", "avg_bw_load.png"),
    ]

    for (log_file, ylabel, fig_name) in metrics:
        pickle_path = os.path.join(base_log_dir, log_file)
        save_path = os.path.join(base_fig_dir, fig_name)
        plot_metric(pickle_path, ylabel, save_path, window=100)
