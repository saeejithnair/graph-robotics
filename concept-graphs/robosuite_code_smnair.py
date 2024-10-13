import tyro
from dataclasses import dataclass
import imageio
import numpy as np
import robosuite.macros as macros
from robosuite import make
import robosuite
from robosuite.controllers import load_controller_config
import os
from PIL import Image
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_transform_matrix
from robosuite.utils.camera_utils import transform_from_pixels_to_world 
import open3d as o3d

# Set the image convention to opencv
macros.IMAGE_CONVENTION = "opencv"

# Set the MUJOCO_GL environment variable to use a software renderer
os.environ["MUJOCO_GL"] = "osmesa"

# def bilinear_interpolate(im, x, y):
#     """
#     Bilinear sampling for pixel coordinates x and y from source image im.
#     Taken from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)

#     x0 = np.floor(x).astype(int)
#     x1 = x0 + 1
#     y0 = np.floor(y).astype(int)
#     y1 = y0 + 1

#     x0 = np.clip(x0, 0, im.shape[1] - 1)
#     x1 = np.clip(x1, 0, im.shape[1] - 1)
#     y0 = np.clip(y0, 0, im.shape[0] - 1)
#     y1 = np.clip(y1, 0, im.shape[0] - 1)

#     Ia = im[y0, x0]
#     Ib = im[y1, x0]
#     Ic = im[y0, x1]
#     Id = im[y1, x1]

#     wa = (x1 - x) * (y1 - y)
#     wb = (x1 - x) * (y - y0)
#     wc = (x - x0) * (y1 - y)
#     wd = (x - x0) * (y - y0)

#     return wa * Ia + wb * Ib + wc * Ic + wd * Id
# def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform):
#     """
#     Helper function to take a batch of pixel locations and the corresponding depth image
#     and transform these points from the camera frame to the world frame.

#     Args:
#         pixels (np.array): pixel coordinates of shape [..., 2]
#         depth_map (np.array): depth images of shape [..., H, W, 1]
#         camera_to_world_transform (np.array): 4x4 Tensor to go from pixel coordinates to world
#             coordinates.

#     Return:
#         points (np.array): 3D points in robot frame of shape [..., 3]
#     """

#     # make sure leading dimensions are consistent
#     pixels_leading_shape = pixels.shape[:-2]
#     depth_map_leading_shape = depth_map.shape[:-3]
#     assert depth_map_leading_shape == pixels_leading_shape

#     # sample from the depth map using the pixel locations with bilinear sampling
#     pixels = pixels.astype(float)
#     im_h, im_w = depth_map.shape[-2:]
#     depth_map_reshaped = depth_map.reshape(-1, im_h, im_w, 1)
#     z = bilinear_interpolate(im=depth_map_reshaped, x=pixels[..., 1:2], y=pixels[..., 0:1])
#     z = z.reshape(*depth_map_leading_shape, 1)  # shape [..., 1]

#     # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
#     # (note that we need to swap the first 2 dimensions of pixels to go from pixel indices
#     # to camera coordinates)
#     cam_pts = [pixels[..., 1:2] * z, pixels[..., 0:1] * z, z, np.ones_like(z)]
#     cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

#     # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
#     mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
#     cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
#     points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
#     return points[..., :3]

@dataclass
class Args:
    environment: str = "Stack"
    robots: str = "Panda"
    camera: str = "frontview" # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').
    video_path: str = "panda_stack_video"
    timesteps: int = 5
    height: int = 512
    width: int = 512
    skip_frame: int = 1
    
controller_configs = load_controller_config(default_controller="OSC_POSE")

def save_img(img, name='test', normalize=False):
    if not normalize:
        im = Image.fromarray(img)
        im.save(name + '.png')
    else:
        im = Image.fromarray(np.uint8(255*img/np.max(img)))
        im.save(name + '.png')
def save_point_cloud(point_cloud, output_path):
    """
    Saves a point cloud to a file using Open3D.

    Args:
        point_cloud (open3d.geometry.PointCloud): The point cloud to save.
        output_path (str): Path to the output file.
    """

    o3d.io.write_point_cloud(output_path, point_cloud)

def main(args: Args):
    # Initialize environment with offscreen renderer
    env = make(
        args.environment,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=args.camera,
        camera_heights=args.height,
        camera_widths=args.width,
        controller_configs=controller_configs,
        camera_depths=True,
        camera_segmentations='instance'
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
        # print(f"Gripper position: {gripper_pos}")
        # print(f"Gripper orientation (quaternion): {gripper_quat}")
        # print(f"{obs.keys()}")

        # Print object positions and poses
        # for obj_name in ["cubeA", "cubeB"]:
        #     if f"{obj_name}_pos" in obs and f"{obj_name}_quat" in obs:
        #         obj_pos = obs[f"{obj_name}_pos"]
        #         obj_quat = obs[f"{obj_name}_quat"]
        # print(f"{obj_name} position: {obj_pos}")
        # print(f"{obj_name} orientation (quaternion): {obj_quat}")

        # Dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"]
            frame = np.uint8(frame * 255)
            writer.append_data(frame)
            print(f"Saving frame #{i}")
            
            real_depth_map = get_real_depth_map(env.sim, obs['frontview_depth'])
            camera_to_world_transform = get_camera_transform_matrix(env.sim, 'frontview', 512, 512)
            pixels = np.moveaxis(np.stack(np.where(obs['frontview_segmentation_instance'] == 1))[:-1], 0, 1) 
            point_cloud = transform_from_pixels_to_world(
                pixels, 
                real_depth_map, 
                camera_to_world_transform)
            save_img(obs['frontview_depth'].reshape((512,512)), 'frontview_depth', normalize = True)

        if done:
            break

    env.close()
    writer.close()
    print(f"Video saved to {video_path}")


if __name__ == "__main__":
    tyro.cli(main)
