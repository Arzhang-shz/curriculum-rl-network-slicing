import os
from stable_baselines3 import PPO
from training.make_env import make_vec_env_from_config

class CurriculumScheduler:
    """
    Orchestrates multi‐stage (curriculum) PPO training. For each stage:
      1. It creates a vectorized environment from a given YAML config.
      2. If it's the first stage, instantiates a new PPO model; otherwise, loads the previous checkpoint.
      3. Trains for a specified number of timesteps.
      4. Saves a checkpoint for that stage.
    """

    def __init__(
        self,
        stages: list,
        timesteps_per_stage: list,
        save_dir: str = "results/models",
        policy_kwargs: dict = None,
        verbose: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        device: str = "auto",
    ):
        """
        Args:
            stages: List of tuples [(config_path_stage1, num_envs_stage1),
                                      (config_path_stage2, num_envs_stage2), ...]
            timesteps_per_stage: List of ints specifying timesteps for each stage.
            save_dir: Directory to save model checkpoints.
            policy_kwargs: Optional dict to pass to PPO (e.g., network architecture).
            verbose: Verbosity level for PPO.
            learning_rate: Learning rate for PPO.
            batch_size: Batch size for PPO.
            device: 'cpu', 'cuda', or 'auto' for PPO.
        """
        assert len(stages) == len(timesteps_per_stage), \
            "Number of stages must match number of timesteps entries."

        self.stages = stages
        self.timesteps = timesteps_per_stage
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

    def run(self):
        """
        Execute the curriculum: train PPO sequentially for each stage.
        Returns:
            The final PPO model (after the last stage).
        """
        model = None

        for idx, ((config_path, num_envs), timesteps) in enumerate(zip(self.stages, self.timesteps), start=1):
            print(f"\n===== Starting Curriculum Stage {idx} =====")
            print(f"Config: {config_path} | Num Env: {num_envs} | Timesteps: {timesteps}")
            # Create vectorized environment for this stage
            vec_env = make_vec_env_from_config(config_path, num_envs)

            # If first stage, create a new PPO model; else, load from previous checkpoint
            if model is None:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    verbose=self.verbose,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    device=self.device,
                    policy_kwargs=self.policy_kwargs
                )
            else:
                # Re‐use the same PPO object and set its env to the new stage's vec_env
                model.set_env(vec_env)

            # Train
            model.learn(total_timesteps=timesteps)

            # Save checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"ppo_stage{idx}.zip")
            model.save(checkpoint_path)
            print(f"Checkpoint for Stage {idx} saved to: {checkpoint_path}")

        print("\n===== Curriculum Training Complete =====")
        return model
