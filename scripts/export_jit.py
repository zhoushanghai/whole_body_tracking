import argparse
import os
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from copy import deepcopy


class DummyEnv:
    def __init__(self, obs_dim, action_dim, device):
        self.num_obs = obs_dim
        self.num_privileged_obs = obs_dim  # Dummy
        self.num_actions = action_dim
        self.device = torch.device(device)
        self.num_envs = 1


def export_policy_as_jit(policy, normalizer, path, filename, obs_dim, device):
    class ExportPolicy(torch.nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, observations):
            if self.normalizer is not None:
                observations = self.normalizer(observations)
            return self.policy(observations)

    export_policy = ExportPolicy(policy, normalizer).eval()
    example_obs = torch.randn(1, obs_dim, device=device)
    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(export_policy, example_obs)
    traced.save(os.path.join(path, filename))
    print(f"[INFO] Exported to {os.path.join(path, filename)}")


def main():
    parser = argparse.ArgumentParser(description="Export BeyondMimic checkpoint to JIT")
    parser.add_argument(
        "--load_run", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_path", type=str, default="exported", help="Output directory"
    )
    parser.add_argument(
        "--obs_dim", type=int, default=160, help="Observation dimension"
    )
    parser.add_argument("--action_dim", type=int, default=29, help="Action dimension")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Matching BeyondMimic config
    agent_cfg = {
        "device": args.device,
        "num_steps_per_env": 48,
        "max_iterations": 30000,
        "save_interval": 500,
        "experiment_name": "g1_flat",
        "empirical_normalization": True,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 2,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    env = DummyEnv(args.obs_dim, args.action_dim, args.device)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args.device)

    print(f"[INFO] Loading checkpoint from {args.load_run}")
    checkpoint = torch.load(args.load_run, map_location=args.device)

    # Load policy and normalizer
    runner.alg.policy.load_state_dict(checkpoint["model_state_dict"])
    if "obs_normalizer" in checkpoint and checkpoint["obs_normalizer"] is not None:
        runner.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        normalizer = runner.obs_normalizer
        print("[INFO] Loaded observation normalizer")
    else:
        normalizer = None
        print("[WARN] No observation normalizer found in checkpoint")

    # Extract metadata for MuJoCo config
    from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG

    joint_names = (
        G1_CYLINDER_CFG.spawn.joint_names
    )  # This might not be right, let's use the one from the runner if possible
    # In Isaac Lab, the order is from the articulation data.
    # But since we are in a dummy env, we can just use the G1_CYLINDER_CFG order.

    print("\n" + "=" * 50)
    print("METADATA FOR MUJOCO CONFIG")
    print("=" * 50)

    # We need the joint names in the order they appear in the policy
    # For G1, it's usually:
    target_joint_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    print("policy_joints:")
    for j in target_joint_names:
        print(f"  - {j}")

    print("\naction_scales:")
    scales = [float(G1_ACTION_SCALE.get(j, 0.25)) for j in target_joint_names]
    print(f"  {scales}")

    print("\ndefault_angles:")
    # Get from G1_CYLINDER_CFG.init_state.joint_pos
    default_joint_pos = G1_CYLINDER_CFG.init_state.joint_pos
    angles = []
    for j in target_joint_names:
        angle = 0.0
        for pattern, val in default_joint_pos.items():
            if pattern.replace(".*", "") in j:
                angle = val
                break
        angles.append(float(angle))
    print(f"  {angles}")
    print("=" * 50 + "\n")

    export_policy_as_jit(
        runner.alg.policy.actor,
        normalizer,
        args.output_path,
        "policy.pt",
        args.obs_dim,
        args.device,
    )


if __name__ == "__main__":
    main()
