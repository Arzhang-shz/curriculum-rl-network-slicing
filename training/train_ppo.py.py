import argparse
import sys
from training.scheduler import CurriculumScheduler

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Curriculum PPO model for Multi‐Domain Slice Placement"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/models",
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--configs",
        nargs=3,
        required=True,
        help="Three YAML config paths (stage1.yaml stage2.yaml stage3.yaml)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=3,
        help="Number of parallel environments per stage",
    )
    parser.add_argument(
        "--timesteps",
        nargs=3,
        type=int,
        default=[2500000, 2500000, 2500000],
        help="Timesteps for each of the three stages",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for PPO",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run PPO on ('cpu', 'cuda', or 'auto')",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Validate that we have exactly 3 configs and 3 timesteps
    if len(args.configs) != 3 or len(args.timesteps) != 3:
        print("Error: Exactly 3 configs and 3 timesteps must be provided.")
        sys.exit(1)

    # Prepare the stages list: [(config1, num_envs), (config2, num_envs), (config3, num_envs)]
    stages = [(args.configs[i], args.num_envs) for i in range(3)]
    timesteps = args.timesteps

    scheduler = CurriculumScheduler(
        stages=stages,
        timesteps_per_stage=timesteps,
        save_dir=args.save_dir,
        verbose=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    scheduler.run()

if __name__ == "__main__":
    main()
