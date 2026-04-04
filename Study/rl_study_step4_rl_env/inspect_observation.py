from __future__ import annotations

from env import SharedControlEnv


def main() -> None:
    env = SharedControlEnv()
    obs, info = env.reset(seed=0)

    print("Initial observation vector")
    for name, value in zip(env.observation_names, obs.tolist()):
        print(f"  {name:>20s}: {value:+.4f}")

    print("\nInterpretation notes")
    print("- action is 1D in [-1, 1]")
    print("- lambda update is lam_next = clip(lam + lambda_rate_max * dt * action, 0, 1)")
    print("- no safety filter is used in this step")

    action = -0.5
    obs_next, reward, terminated, truncated, info_next = env.step(action)
    print("\nAfter one step with action = -0.5")
    print(f"  lambda prev -> next: {info_next['lam_prev']:.3f} -> {info_next['lam_next']:.3f}")
    print(f"  delta_drv_rwa      : {info_next['delta_drv_rwa']:.4f} rad")
    print(f"  delta_auto_rwa     : {info_next['delta_auto_rwa']:.4f} rad")
    print(f"  delta_cmd          : {info_next['delta_cmd']:.4f} rad")
    print(f"  reward             : {reward:.4f}")
    print(f"  terminated         : {terminated}")
    print(f"  truncated          : {truncated}")


if __name__ == "__main__":
    main()
