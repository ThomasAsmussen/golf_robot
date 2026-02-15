import json
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load JSONL and keep only first entry per timestamp
# -------------------------------------------------
here = Path(__file__).parents[2]
jsonl_path = here / "log" / "real_episodes" / "episode_logger.jsonl"
print(f"Loading data from {jsonl_path}")

seen_times = set()
xs, ys = [], []
angles = []

with jsonl_path.open("r") as f:
    for line in f:
        obj = json.loads(line)
        t = obj["time"]

        if t in seen_times:
            continue
        seen_times.add(t)

        x, y = obj["ball_start_obs"]
        angle = obj["angle_deg"]
        xs.append(x)
        ys.append(y)
        angles.append(angle)

print(f"Loaded {len(xs)} unique initial positions")

# -------------------------------------------------
# Scatter plot of initial positions
# -------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(xs, ys, s=10)
plt.xlabel("ball_start_obs x [m]")
plt.ylabel("ball_start_obs y [m]")
plt.title("Initial ball positions (unique timestamps)")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------------------------
# Marginal distributions
# -------------------------------------------------
plt.figure(figsize=(6, 3))
plt.hist(xs, bins=50)
plt.xlabel("ball_start_obs x [m]")
plt.ylabel("count")
plt.title("Distribution of x (unique timestamps)")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(6, 3))
plt.hist(ys, bins=50)
plt.xlabel("ball_start_obs y [m]")
plt.ylabel("count")
plt.title("Distribution of y (unique timestamps)")
plt.grid(True, alpha=0.3)
plt.show()

# plt.figure(figsize=(6, 3))
# plt.hist(angles, bins=50)
# plt.xlabel("angle [deg]")
# plt.ylabel("count")
# plt.title("Distribution of angles (unique timestamps)")
# plt.grid(True, alpha=0.3)
# plt.show()