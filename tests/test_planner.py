import os
import sys
import importlib
import numpy as np

# ensure package src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# map package-local modules to top-level names because many modules use
# legacy top-level imports like `from config import ...`.
config_module = importlib.import_module('golf_robot.planning.config')
utils_module = importlib.import_module('golf_robot.planning.utils')
kin_module = importlib.import_module('golf_robot.planning.kinematics')
import sys as _sys
_sys.modules['config'] = config_module
_sys.modules['utils'] = utils_module
_sys.modules['kinematics'] = kin_module

from golf_robot.planning import trajectory


def test_generate_trajectory_basic():
    """generate_trajectory should return a results dict with expected keys for a small, feasible speed."""
    impact_speed = 0.3  # small speed to increase feasibility
    impact_angle = 0.0  # along +X

    results = trajectory.generate_trajectory(impact_speed, impact_angle)
    assert isinstance(results, dict)

    # keys and basic contents
    for k in ('t_plan', 'P_plan', 'Q_plan', 'dQ_plan', 'ddQ_plan', 'impact_sample_idx'):
        assert k in results

    t = results['t_plan']
    Q = results['Q_plan']
    dQ = results['dQ_plan']

    assert len(t) > 0
    assert Q.shape[0] == len(t)
    assert dQ.shape[0] == len(t)

    # achieved impact speed should be finite and non-negative
    if 'achieved_impact_speed' in results:
        s = results['achieved_impact_speed']
        assert np.isfinite(s)
        assert s >= 0.0


def test_tcp_path_from_Q_consistency():
    """tcp_path_from_Q should return TCP positions consistent with fk_ur10 for each row in Q."""
    # reuse a known reachable joint config
    q_ref = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])
    Q = np.vstack([q_ref, q_ref])

    P = trajectory.tcp_path_from_Q(Q)
    assert P.shape == (2, 3)

    # compare with fk_ur10 directly
    fk = kin_module.fk_ur10(q_ref)[-1][:3, 3]
    assert np.allclose(P[0], fk)
    assert np.allclose(P[1], fk)


def test_joint_vel_from_tcp_feasible():
    """joint_vel_from_tcp should return feasible dq for a moderate TCP velocity at a typical pose."""
    q = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])
    v_lin = np.array([0.1, 0.0, 0.0])  # small desired TCP linear velocity

    dq, v_tcp, feasible = trajectory.joint_vel_from_tcp(q, v_lin)
    assert dq.shape == (6,)
    assert v_tcp.shape == (3,)
    assert isinstance(feasible, (bool, np.bool_))
