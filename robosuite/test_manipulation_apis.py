import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import os
import robosuite.macros as macros
from manipulation import SimEnv, Pose

# Set the image convention to opencv
macros.IMAGE_CONVENTION = "opencv"

# Set the MUJOCO_GL environment variable to use a software renderer
os.environ["MUJOCO_GL"] = "osmesa"


def run_atomic_tasks(env, render=True):
    sim_env = SimEnv(env, render)

    tasks = [
        # test_open_close_gripper,
        test_move_to_position,
        test_rotate_gripper,
        test_tilt_gripper,
        test_align_gripper,
        test_approach_object,
        test_pick_and_place,
        test_trajectory,
    ]

    for task in tasks:
        print(f"Running task: {task.__name__}")
        sim_env.reset()
        task(sim_env, render)
        input("Press Enter to continue to the next task...")


def test_open_close_gripper(sim_env, render):
    print("Opening gripper...")
    sim_env.open_gripper(width=1.0)
    print("Closing gripper...")
    sim_env.close_gripper()
    print("Opening gripper...")
    sim_env.open_gripper(width=1.0)


def test_move_to_position(sim_env, render):
    print("Moving to a target pose...")
    # target_pose = Pose(pos=np.array([0.0, 0.3, 0.3]), quat=np.array([1, 0, 0, 0]))
    poses = sim_env.get_all_object_poses()
    print(poses)
    target_pose = poses["cube"]
    target_pose.pos[2]
    print(f"Target pose: {target_pose}")
    sim_env.move_to_position(target_pose)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_rotate_gripper(sim_env, render):
    print("Rotating gripper around z-axis...")
    current_pose = sim_env.get_gripper_pose()
    rotated_quat = sim_env.rotate_around_gripper_z_axis(90)
    sim_env.move_to_position(Pose(current_pose.pos, None))
    sim_env.rotate_to_orientation(rotated_quat)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_tilt_gripper(sim_env, render):
    print("Tilting gripper up and down...")
    current_pose = sim_env.get_gripper_pose()
    tilted_quat = sim_env.tilt_updown(30)
    sim_env.move_to_position(Pose(current_pose.pos, tilted_quat))
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))

    print("Tilting gripper left and right...")
    tilted_quat = sim_env.tilt_leftright(30)
    sim_env.move_to_position(Pose(current_pose.pos, tilted_quat))
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_align_gripper(sim_env, render):
    print("Aligning gripper vertically...")
    current_pose = sim_env.get_gripper_pose()
    vertical_quat = sim_env.get_vertical_ori()
    sim_env.move_to_position(Pose(current_pose.pos, vertical_quat))
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))

    print("Aligning gripper horizontally...")
    horizontal_quat = sim_env.get_horizontal_ori()
    sim_env.move_to_position(Pose(current_pose.pos, horizontal_quat))
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_approach_object(sim_env, render):
    print("Approaching an object...")
    sim_env.approach_object("cubeA", distance=0.05)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_pick_and_place(sim_env, render):
    print("Performing a pick and place task...")

    # Approach the object
    sim_env.approach_object("cubeA", distance=0.05)

    # Align gripper with the object
    object_pose = sim_env.get_object_pose("cubeA")
    sim_env.move_to_position(Pose(object_pose.pos, sim_env.get_vertical_ori()))

    # Move to grasp position
    grasp_pose = Pose(
        object_pose.pos + np.array([0, 0, 0.02]), sim_env.get_vertical_ori()
    )
    sim_env.move_to_position(grasp_pose)

    # Close gripper to grasp object
    sim_env.close_gripper(speed=0.05)

    # Lift object
    sim_env.lift_object(height=0.3)

    # Move to target position
    target_pos = np.array([0.3, -0.2, 0.3])
    sim_env.move_to_position(Pose(target_pos, sim_env.get_vertical_ori()))

    # Place object
    sim_env.place_object(target_pos - np.array([0, 0, 0.05]))

    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_trajectory(sim_env, render):
    print("Executing a trajectory...")
    waypoints = [
        Pose(np.array([0.2, 0.2, 0.3]), sim_env.get_vertical_ori()),
        Pose(np.array([-0.2, 0.2, 0.3]), sim_env.get_horizontal_ori()),
        Pose(np.array([-0.2, -0.2, 0.3]), sim_env.get_vertical_ori()),
        Pose(np.array([0.2, -0.2, 0.3]), sim_env.get_horizontal_ori()),
    ]
    sim_env.execute_trajectory(waypoints)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


if __name__ == "__main__":
    # Create the robosuite environment
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        ignore_done=True,
    )

    run_atomic_tasks(env)
    env.close()
