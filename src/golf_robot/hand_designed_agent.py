from turtle import distance
import numpy as np
from rl_common import unscale_state_vec

def hand_tuned_policy(state_vec: np.ndarray) -> np.ndarray:
    """
    A simple hand-designed agent for the golf robot that aims directly at the hole
    with a fixed power.

    Args:
        observation (np.ndarray): The observation from the environment, which includes
                                  the position of the ball and the hole.
    Returns:
        np.ndarray: The action to be taken by the agent.
    """
    # Extract ball and hole positions from the observation
    ball_position = state_vec[0:2]  # Assuming first two elements are ball x, y
    hole_position = state_vec[2:4]  # Assuming next two elements are hole x, y
    # unscale_obs = unscale_state_vec(state_vec)
    # ball_position = unscale_obs[0:2]
    # hole_position = unscale_obs[2:4]

    # Calculate the direction vector from ball to hole
    direction = hole_position - ball_position
    
    angle_rad = np.arctan2(direction[1], direction[0])
    angle_deg = np.degrees(angle_rad)

    # Define a fixed power for the shot
    distance = np.linalg.norm(direction)
    # speed = min(distance * 0.55, 2.0)  # Scale power based on distance, capped at 1.8   
    speed = min(distance * 0.44, 1.8)
    return speed, angle_deg