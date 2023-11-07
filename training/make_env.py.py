import yaml
from stable_baselines3.common.vec_env import DummyVecEnv
from env.placement_env import PlacementEnv

def make_env_factory(config_path: str):
    """
    Returns a callable that, when called, creates one instance of PlacementEnv
    using the YAML config at 'config_path'.
    This is intended for use with SB3's VecEnv utilities (e.g., DummyVecEnv).
    """

    def _init():
        # Load configuration from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Construct and return the environment
        env = PlacementEnv(config)
        return env

    return _init

def make_vec_env_from_config(config_path: str, num_envs: int = 1):
    """
    Utility to create a DummyVecEnv with `num_envs` parallel copies of PlacementEnv
    each initialized from the same YAML config file.
    """
    env_fns = [make_env_factory(config_path) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    return vec_env
