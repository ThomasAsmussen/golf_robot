# planning/warmstart_knn.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise ImportError(
        "warmstart_knn.py requires scikit-learn. Install with: pip install scikit-learn"
    ) from e


@dataclass
class WarmstartKNN:
    scaler: StandardScaler
    nn: NearestNeighbors
    X: np.ndarray                 # (N,4) raw features
    start_off: np.ndarray         # (N,3)
    end_off: np.ndarray           # (N,3)
    J: np.ndarray                 # (N,)

    def query(
        self,
        impact_speed: float,
        impact_angle_deg: float,
        ball_x_offset: float,
        ball_y_offset: float,
        *,
        k: int = 15,
        n_keep: int = 8,
        prefer_low_J: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns up to n_keep (start_off, end_off) pairs from nearest neighbors.
        """
        q = np.array([[impact_speed, impact_angle_deg, ball_x_offset, ball_y_offset]], dtype=float)
        qz = self.scaler.transform(q)

        k = int(min(k, self.X.shape[0]))
        dists, idxs = self.nn.kneighbors(qz, n_neighbors=k, return_distance=True)
        idxs = idxs[0]

        # Optionally re-rank neighbors by J (while still staying local in feature space)
        if prefer_low_J:
            idxs = idxs[np.argsort(self.J[idxs])]

        # Dedup offsets (because your JSONL often repeats same condition/seed)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        seen = set()
        for i in idxs:
            so = np.asarray(self.start_off[i], float)
            eo = np.asarray(self.end_off[i], float)
            key = (tuple(np.round(so, 4)), tuple(np.round(eo, 4)))
            if key in seen:
                continue
            seen.add(key)
            out.append((so, eo))
            if len(out) >= n_keep:
                break
        return out


def load_warmstart_knn(jsonl_path: str | Path) -> WarmstartKNN:
    """
    Reads speed_angle_grid_start_end_all.jsonl and builds a KNN index.

    Keeps only:
      - type == "record"
      - ok == True
      - has start_off/end_off
    If multiple records share identical (speed,angle,bx,by), keep the lowest-J one.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)

    best_by_key = {}  # key -> (J, feat4, start_off3, end_off3)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("type") != "record":
                continue
            if not bool(rec.get("ok", False)):
                continue
            if ("start_off" not in rec) or ("end_off" not in rec):
                continue

            feat = (
                float(rec["impact_speed"]),
                float(rec["impact_angle_deg"]),
                float(rec["ball_x_offset"]),
                float(rec["ball_y_offset"]),
            )
            key = tuple(np.round(np.array(feat), 6))  # stable dedup key
            J = float(rec.get("J", rec.get("peak_abs_ddq", 1e9)))

            start_off = np.asarray(rec["start_off"], dtype=float)
            end_off = np.asarray(rec["end_off"], dtype=float)

            prev = best_by_key.get(key, None)
            if (prev is None) or (J < prev[0]):
                best_by_key[key] = (J, feat, start_off, end_off)

    if not best_by_key:
        raise RuntimeError(f"No usable warm-start records found in {path}")

    Js, feats, starts, ends = zip(*best_by_key.values())
    X = np.asarray(feats, dtype=float)
    start_off = np.asarray(starts, dtype=float)
    end_off = np.asarray(ends, dtype=float)
    J = np.asarray(Js, dtype=float)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=min(50, len(Xz)), algorithm="auto")
    nn.fit(Xz)

    return WarmstartKNN(
        scaler=scaler,
        nn=nn,
        X=X,
        start_off=start_off,
        end_off=end_off,
        J=J,
    )