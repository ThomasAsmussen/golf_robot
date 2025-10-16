import math

def estimate_crr(v0, s, g=9.81):
    """Estimate rolling resistance coefficient from rollout distance."""
    return (v0**2) / (2.0 * g * s)

def suggest_mujoco_values(crr_hat, ball_guess=0.003):
    """
    Given an estimated Crr and a guess for the ball's rolling friction,
    compute the required green rolling friction so that
    ball * green ≈ Crr.
    """
    green_val = crr_hat / ball_guess
    return ball_guess, green_val

def main():
    # Inputs
    v0 = 1.0   # initial speed [m/s]
    s  = 2.0   # stopping distance [m]

    # Step 1: estimate effective rolling resistance
    crr_hat = estimate_crr(v0, s)
    print(f"Estimated rolling resistance coefficient Crr: {crr_hat:.6f}")

    # Step 2: suggest ball/green friction split for MuJoCo
    ball_val, green_val = suggest_mujoco_values(crr_hat, ball_guess=0.003)
    print(f"Suggested MuJoCo rolling frictions:")
    print(f"  Ball  (3rd friction term):  {ball_val:.6f}")
    print(f"  Green (3rd friction term): {green_val:.6f}")
    print(f"  Effective product: {ball_val * green_val:.6f}")

if __name__ == "__main__":
    main()
