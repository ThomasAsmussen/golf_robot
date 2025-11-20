import os
import sys
import numpy as np

# Make sure the 'src' directory is on sys.path so we can import the module under test
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from golf_robot.planning import kinematics


def test_fk_returns_transforms_shape():
    """fk_ur10 should return 10 homogeneous transforms of shape (4,4)."""
    q = np.zeros(6)
    Ts = kinematics.fk_ur10(q)
    assert isinstance(Ts, list)
    assert len(Ts) == 10
    for T in Ts:
        assert T.shape == (4, 4)


def test_fk_ik_roundtrip_seed_returns_same_q():
    """Using a known joint configuration as seed, IK should return that configuration."""
    # example configuration used elsewhere in the project (known reachable pose)
    q_ref = np.array([-2.18480298, -2.68658532, -1.75772109,  1.30184109,  0.61400683,  0.50125856])
    T_tcp = kinematics.fk_ur10(q_ref)[-1]
    q_sol, info = kinematics.pick_ik_solution(T_tcp, q_ref)
    assert q_sol is not None, "pick_ik_solution failed to return a solution"
    # the returned solution should be numerically close to the seed
    assert np.allclose(q_sol, q_ref, atol=1e-6)


def test_numeric_jacobian_finite_difference():
    """numeric_jacobian * dq should approximate the finite-difference twist for small dq."""
    np.random.seed(0)
    q = np.array([-1.0, -2.0, -1.0, 1.0, 0.5, 0.2])
    dq_vel = np.random.randn(6)
    eps = 1e-6

    # numerical Jacobian from module
    J = kinematics.numeric_jacobian(q)

    # Finite-difference approximation of twist for q -> q + dq_vel * eps
    T0 = kinematics.fk_ur10(q)[-1]
    T1 = kinematics.fk_ur10(q + dq_vel * eps)[-1]

    p0, p1 = T0[:3, 3], T1[:3, 3]
    R0, R1 = T0[:3, :3], T1[:3, :3]

    v_fd = (p1 - p0) / eps
    dR = R1 @ R0.T
    w_fd = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2.0 * eps)

    twist_fd = np.hstack((v_fd, w_fd))

    twist_j = J @ dq_vel

    # Allow some tolerance since both sides are numerically approximated
    assert np.allclose(twist_fd, twist_j, atol=1e-4, rtol=1e-3)
