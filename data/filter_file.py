import pandas as pd
from pathlib import Path

# --- edit this to your input file ---
in_path = Path("data/tuning_videos_processed/test_0.6-1_trajectory.csv")
out_path = in_path.with_name(in_path.stem + "_time_shifted.csv")

# aliases for auto-detection
aliases = {
    "time": ["time","t","timestamp","Time","TIME","Time[s]","time_s","time(sec)"],
}

def pick(colset, names):
    for n in names:
        if n in colset:
            return n
    return None

# read data
df = pd.read_csv(in_path)
cols = set(df.columns)

# find time column
time_col = pick(cols, aliases["time"])
if time_col is None:
    raise ValueError(
        f"Missing required time column (tried common aliases): {aliases['time']}\n"
        f"Found columns: {list(df.columns)}"
    )

# shift time so it starts at 0
t = df[time_col].to_numpy()
df[time_col] = t - 4.3

# save result
df.to_csv(out_path, index=False)
print(f"Saved time-shifted file to: {out_path}")
