import tyro
from dataclasses import dataclass
import imageio
import numpy as np
import robosuite.macros as macros
from robosuite import make
import os

# Set the image convention to opencv
macros.IMAGE_CONVENTION = "opencv"

# Set the MUJOCO_GL environment variable to use a software renderer
os.environ["MUJOCO_GL"] = "osmesa"

@dataclass
class Args:
    environment: str = "Stack"
    robots: str = "Panda"
    camera: str = "frontview"
    video_path: str = "panda_stack_video"
    timesteps: int = 500
    height: int = 512
    width: int = 512
    skip_frame: int = 1

def main(args: Args):
    # Initialize environment with offscreen renderer
    env = make(
        args.environment,
        robots=args.robots,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    obs = env.reset()
    ndim = env.action_dim

    # Print available observation keys
    print("Available observation keys:", obs.keys())

    # Create a video writer with imageio
    video_path = f"{args.video_path}_{args.camera}.mp4"
    writer = imageio.get_writer(video_path, fps=20)

    for i in range(args.timesteps):
        # Run a uniformly random agent
        action = np.random.randn(ndim)
        obs, reward, done, info = env.step(action)

        # Print gripper position and pose
        gripper_pos = obs["robot0_eef_pos"]
        gripper_quat = obs["robot0_eef_quat"]
        print(f"Gripper position: {gripper_pos}")
        print(f"Gripper orientation (quaternion): {gripper_quat}")
        # print(f"{obs.keys()}")

        # Print object positions and poses
        for obj_name in ["cubeA", "cubeB"]:
            if f"{obj_name}_pos" in obs and f"{obj_name}_quat" in obs:
                obj_pos = obs[f"{obj_name}_pos"]
                obj_quat = obs[f"{obj_name}_quat"]
                print(f"{obj_name} position: {obj_pos}")
                print(f"{obj_name} orientation (quaternion): {obj_quat}")

        # Dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"]
            writer.append_data(frame)
            print(f"Saving frame #{i}")

        if done:
            break

    env.close()
    writer.close()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    tyro.cli(main)