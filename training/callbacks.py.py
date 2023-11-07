from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EpisodeMetricsLogger(BaseCallback):
    """
    A simple callback to log episode reward, acceptance ratio, and load percentage
    after each episode. These metrics will be stored in a list that can be
    later pickled or used for plotting.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Containers for aggregated metrics
        self.episode_rewards = []
        self.episode_acceptances = []
        self.episode_loads = []

    def _on_step(self) -> bool:
        # Check if an episode just ended in any of the parallel envs
        done_array = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if done_array is not None and infos is not None:
            # In SB3, 'infos' is a list (or tuple) of dicts—one per environment
            for done, info in zip(done_array, infos):
                if done:
                    # The environment should have logged these into 'info'
                    # e.g. info = {'episode_reward': ..., 'acceptance_ratio': ..., 'load_pct': ...}
                    ep_reward = info.get("episode_reward", None)
                    ep_accept = info.get("acceptance_ratio", None)
                    ep_load = info.get("load_pct", None)

                    if ep_reward is not None:
                        self.episode_rewards.append(ep_reward)
                    if ep_accept is not None:
                        self.episode_acceptances.append(ep_accept)
                    if ep_load is not None:
                        self.episode_loads.append(ep_load)
        return True

    def _on_training_end(self) -> None:
        # At the end of training, make these available on the callback instance
        self.model.episode_rewards = np.array(self.episode_rewards)
        self.model.episode_acceptances = np.array(self.episode_acceptances)
        self.model.episode_loads = np.array(self.episode_loads)
