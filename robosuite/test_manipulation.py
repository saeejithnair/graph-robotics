import numpy as np
from robosuite import make
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_robot_state(obs):
    gripper_pos = obs["robot0_eef_pos"]
    gripper_quat = obs["robot0_eef_quat"]
    logger.info(f"Gripper position: {gripper_pos}")
    logger.info(f"Gripper orientation (quaternion): {gripper_quat}")

    gripper_state = obs["robot0_gripper_qpos"]
    logger.info(f"Gripper state: {gripper_state}")
    logger.info(
        f"Gripper is holding object: {np.any(np.abs(gripper_state) < 0.01)}"
    )  # Assumes gripper is closed if any finger is < 0.045

    logger.info(f"CubeA position: {obs['cubeA_pos']}")
    logger.info(f"CubeB position: {obs['cubeB_pos']}")


def create_move_action(env, obs, target_pos):
    current_pos = obs["robot0_eef_pos"]
    action = np.zeros(env.action_dim)
    action[:3] = (target_pos - current_pos) * 5  # Proportional control
    return action


def create_grasp_action(env):
    action = np.zeros(env.action_dim)
    action[-1] = (
        -1
    )  # Close gripper (assumes last action component controls the gripper)
    return action


def create_release_action(env):
    action = np.zeros(env.action_dim)
    action[-1] = 1  # Open gripper (assumes last action component controls the gripper)
    return action


def move_to_pose(env, obs, target_pos, threshold=0.01, max_steps=100):
    for _ in range(max_steps):
        action = create_move_action(env, obs, target_pos)
        obs, reward, done, info = env.step(action)
        env.render()

        current_pos = obs["robot0_eef_pos"]
        if np.linalg.norm(current_pos - target_pos) < threshold:
            return obs, True

    logger.warning(
        f"Failed to reach target position {target_pos} after {max_steps} steps"
    )
    return obs, False


def pick_up_object(env, obs, object_pos):
    # Move above the object
    above_pos = object_pos + np.array([0, 0, 0.05])  # 5cm above the object
    obs, success = move_to_pose(env, obs, above_pos)
    if not success:
        return obs, False

    # Move down to the object
    obs, success = move_to_pose(env, obs, object_pos)
    if not success:
        return obs, False

    # Grasp the object
    for _ in range(10):  # Try grasping for 10 steps
        action = create_grasp_action(env)
        obs, reward, done, info = env.step(action)
        env.render()

    # Check if grasp was successful
    if np.any(obs["robot0_gripper_qpos"] < 0.01):
        logger.info("Successfully grasped the object")

        # Lift the object
        lift_pos = object_pos + np.array([0, 0, 0.1])  # 10cm above the object
        obs, success = move_to_pose(env, obs, lift_pos)
        return obs, success
    else:
        logger.error("Failed to grasp the object")
        return obs, False


def place_object(env, obs, target_pos):
    # Move above the target position
    above_pos = target_pos + np.array([0, 0, 0.1])  # 10cm above the target
    obs, success = move_to_pose(env, obs, above_pos)
    if not success:
        return obs, False

    # Move down to the target position
    obs, success = move_to_pose(env, obs, target_pos)
    if not success:
        return obs, False

    # Release the object
    for _ in range(10):  # Try releasing for 10 steps
        action = create_release_action(env)
        obs, reward, done, info = env.step(action)
        env.render()

    # Move up
    obs, _ = move_to_pose(env, obs, above_pos)

    logger.info("Object placed at target position")
    return obs, True


def main():
    env = make(
        "Stack",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
    )

    obs = env.reset()
    logger.info(
        f"Environment reset. Starting stacking task. Action space dimension: {env.action_dim}"
    )

    max_attempts = 10
    for attempt in range(max_attempts):
        logger.info(f"Stacking attempt {attempt + 1}/{max_attempts}")

        # Log initial state
        logger.info("Current robot and environment state:")
        log_robot_state(obs)

        # Pick up cubeA
        cubeA_pos = obs["cubeA_pos"]
        logger.info(f"Attempting to pick up cubeA at position {cubeA_pos}")
        obs, success = pick_up_object(env, obs, cubeA_pos)
        if not success:
            logger.error("Failed to pick up cubeA. Retrying.")
            # obs = env.reset()
            continue

        # Get cubeB position for stacking
        cubeB_pos = obs["cubeB_pos"]
        stack_pos = cubeB_pos + np.array([0, 0, 0.025])  # Stack slightly above cubeB

        # Place cubeA on top of cubeB
        logger.info(f"Attempting to stack cubeA on cubeB at position {stack_pos}")
        obs, success = place_object(env, obs, stack_pos)
        if not success:
            logger.error("Failed to stack cubeA on cubeB. Retrying.")
            # obs = env.reset()
            continue

        # Check if stacking was successful
        cubeA_new_pos = obs["cubeA_pos"]
        if np.linalg.norm(cubeA_new_pos - stack_pos) < 0.02:  # Within 2cm of target
            logger.info("Stacking task completed successfully!")
            break
        else:
            logger.error("Stacking was not precise. Retrying.")
            # obs = env.reset()

    # Final state logging
    logger.info("Final robot and environment state:")
    log_robot_state(obs)

    return env


if __name__ == "__main__":
    env = main()
    env.close()
