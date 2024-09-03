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
        # test_move_to_position,
        # test_rotate_gripper,
        # test_tilt_gripper,
        # test_align_gripper,
        # test_approach_object,
        test_pick_and_place,
        test_trajectory,
    ]

    for task in tasks:
        print(f"Running task: {task.__name__}")
        sim_env.reset()
        task(sim_env, render)
        # input("Press Enter to continue to the next task...")


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
    target_pose.quat = None
    print(f"Target pose: {target_pose}")
    sim_env.move_to_pose(target_pose)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def degs_to_rads(degs):
    return np.deg2rad(degs)


def rads_to_degs(rads):
    return np.rad2deg(rads)


def test_rotate_gripper(sim_env, render):
    print("Rotating gripper around z-axis...")
    current_pose = sim_env.get_gripper_pose()
    rotated_quat = sim_env.rotate_around_gripper_z_axis(89)
    sim_env.move_to_pose(Pose(current_pose.pos, rotated_quat))

    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))
    print("Rotating gripper back around z-axis...")
    current_pose = sim_env.get_gripper_pose()
    rotated_quat = sim_env.rotate_around_gripper_z_axis(-89)
    sim_env.move_to_pose(Pose(current_pose.pos, rotated_quat))
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_tilt_gripper(sim_env, render):
    print("\n--- Testing Gripper Tilt ---")

    def render_steps(steps=100):
        if render:
            print(f"Rendering for {steps} steps...")
            for _ in range(steps):
                sim_env.step(np.zeros(7))

    # Get the initial gripper pose
    initial_pose = sim_env.get_gripper_pose()
    print(
        f"[DEBUG] Initial gripper pose: position = {initial_pose.pos}, orientation = {initial_pose.quat}"
    )

    # Test tilt up
    print("\nTilting gripper up by 30 degrees...")
    tilted_quat = sim_env.tilt_updown(30)
    tilted_pose = Pose(initial_pose.pos, tilted_quat)
    print(f"[DEBUG] Tilted pose (up): {tilted_pose}")
    sim_env.move_to_pose(tilted_pose)
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose after tilt up: {current_pose}")
    render_steps()

    # Undo tilt up
    print("\nReturning gripper to original orientation...")
    sim_env.move_to_pose(initial_pose)
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose after undoing tilt: {current_pose}")
    render_steps()

    # Test tilt left
    print("\nTilting gripper left by 30 degrees...")
    tilted_quat = sim_env.tilt_leftright(30)
    tilted_pose = Pose(initial_pose.pos, tilted_quat)
    print(f"[DEBUG] Tilted pose (left): {tilted_pose}")
    sim_env.move_to_pose(tilted_pose)
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose after tilt left: {current_pose}")
    render_steps()

    # Undo tilt left
    print("\nReturning gripper to original orientation...")
    sim_env.move_to_pose(initial_pose)
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Final gripper pose: {current_pose}")
    render_steps()

    print("--- Gripper Tilt Test Completed ---\n")


def test_align_gripper(sim_env, render):
    print("Aligning gripper vertically...")
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose before vertical alignment: {current_pose}")
    vertical_quat = sim_env.get_vertical_ori()
    print(f"[DEBUG] Vertical orientation quaternion: {vertical_quat}")
    sim_env.move_to_pose(Pose(current_pose.pos, vertical_quat))
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose after vertical alignment: {current_pose}")
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))

    print("Aligning gripper horizontally...")
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose before horizontal alignment: {current_pose}")
    horizontal_quat = sim_env.get_horizontal_ori()
    print(f"[DEBUG] Horizontal orientation quaternion: {horizontal_quat}")
    sim_env.move_to_pose(Pose(current_pose.pos, horizontal_quat))
    current_pose = sim_env.get_gripper_pose()
    print(f"[DEBUG] Current pose after horizontal alignment: {current_pose}")
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_approach_object(sim_env, render):
    print("Approaching an object...")
    object_pose = sim_env.get_object_pose("cube")
    print(f"[DEBUG] Object pose: {object_pose}")
    if object_pose is None:
        print("[ERROR] Object 'cubeA' not found. Skipping approach test.")
        return
    sim_env.approach_object("cube", distance=0.05)
    if render:
        for _ in range(100):
            sim_env.step(np.zeros(7))


def test_pick_and_place(sim_env, render):
    print("Performing a pick and place task...")

    # Approach the object
    sim_env.approach_object("cube", distance=0.05)

    # Align gripper with the object
    object_pose = sim_env.get_object_pose("cube")
    sim_env.move_to_pose(Pose(object_pose.pos, sim_env.get_vertical_ori()))

    # Move to grasp position
    grasp_pose = Pose(
        object_pose.pos + np.array([0, 0, 0.02]), sim_env.get_vertical_ori()
    )
    sim_env.move_to_pose(grasp_pose)

    # Close gripper to grasp object
    sim_env.close_gripper(speed=0.05)

    # Lift object
    sim_env.lift_object(height=0.3)

    # Move to target position
    target_pos = np.array([0.3, -0.2, 0.3])
    sim_env.move_to_pose(Pose(target_pos, sim_env.get_vertical_ori()))

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
