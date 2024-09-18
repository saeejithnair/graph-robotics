from typing import Dict, Optional, Tuple, List
import numpy as np
from robosuite.utils.transform_utils import (
    quat2axisangle,
    axisangle2quat,
    quat_multiply,
    quat_conjugate,
    mat2quat,
    quat2mat,
)
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray


class SimEnv:
    def __init__(self, env, render: bool = False):
        self.env = env
        self.render = render
        self.obs = env.reset()
        self.gripper_width = 0.0
        self.MAX_GRIPPER_WIDTH = 0.04  # Maximum gripper width in meters
        # self.MAX_GRIPPER_WIDTH = 0.08570  # Maximum gripper width in meters
        self.GRIPPER_SPEED = 0.1
        self.GRIPPER_FORCE = 40
        self.FINGERTIP_OFFSET = np.array([0, 0, -0.095])  # Offset from ee to fingertip
        self.HOME_QUAT = np.array([0.9201814, -0.39136365, 0.00602445, 0.00802529])

    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        if self.render:
            self.env.render()

    def null_step(self, num_steps: int = 10):
        for _ in range(num_steps):
            self.step(np.zeros(self.env.action_dim))

    def get_gripper_pose(self) -> Pose:
        gripper_pos = self.obs["robot0_eef_pos"]
        gripper_quat = self.obs["robot0_eef_quat"]
        return Pose(gripper_pos, gripper_quat)

    def get_gripper_width(self) -> float:
        return self.obs["robot0_gripper_qpos"][0]

    def get_object_pose(self, object_id: str) -> Optional[Pose]:
        obj_pos = self.obs.get(f"{object_id}_pos")
        obj_quat = self.obs.get(f"{object_id}_quat")
        if obj_pos is None or obj_quat is None:
            print(f"[WARNING] Object {object_id} not found in the environment.")
            return None
        return Pose(obj_pos, obj_quat)

    def get_all_object_poses(self) -> Dict[str, Pose]:
        object_poses: Dict[str, Pose] = {}
        for obs in self.obs:
            if obs.endswith("_pos"):
                obj_id = obs.split("_")[0]
                object_poses[obj_id] = self.get_object_pose(obj_id)
        return object_poses

    def make_gripper_action(self, open: bool) -> np.ndarray:
        action = np.zeros(self.env.action_dim)
        action[-1] = -1 if open else 1
        return action

    def open_gripper(self, width: float = 1.0, max_steps: int = 100) -> bool:
        target_width = min(width * self.MAX_GRIPPER_WIDTH, self.MAX_GRIPPER_WIDTH)
        for _ in range(
            max_steps
        ):  # Limit the number of steps to prevent infinite loops
            action = self.make_gripper_action(open=True)

            self.step(action)
            self.gripper_width = self.get_gripper_width()

            if abs(self.gripper_width - target_width) <= 0.001:
                return True

        return False

    def close_gripper(self) -> bool:
        max_steps = 100
        while self.gripper_width > 0 and max_steps > 0:
            action = action = self.make_gripper_action(open=False)
            self.step(action)
            self.gripper_width = self.get_gripper_width()
            max_steps -= 1
            if self.gripper_width <= 0.0245:
                return True

        print(f"[DEBUG] Gripper width is {self.gripper_width}")
        return False

    def move_to_pose(self, target_pose: Pose, max_steps: int = 1000) -> bool:
        # Move along each axis
        for i in range(3):  # 0: X, 1: Y, 2: Z
            target_pos = self.get_gripper_pose().pos.copy()
            # print(f"Current position: {target_pos}")
            target_pos[i] = target_pose.pos[i]
            # print(f"Moving to {target_pos}")
            if not self.move_along_axis(target_pos, i, max_steps):
                return False

        # # Rotate to target orientation
        return self.rotate_to_orientation(target_pose.quat, max_steps)

    def move_along_axis(
        self, target_pos: np.ndarray, axis: int, max_steps: int
    ) -> bool:
        for _ in range(max_steps):
            current_pos = self.get_gripper_pose().pos
            direction = target_pos - current_pos

            if np.allclose(current_pos[axis], target_pos[axis], atol=1e-3):
                print(
                    f"Reached target position {current_pos} at step {_} for axis {axis}"
                )
                return True

            action = np.zeros(7)
            action[axis] = direction[axis] * 10
            self.step(action)

        print(f"Failed to move along axis {axis} to {target_pos} from {current_pos}")
        return False

    def rotate_to_orientation(
        self, target_quat: np.ndarray, max_steps: int = 1000
    ) -> bool:
        if target_quat is None:
            return True

        for _ in range(max_steps):
            current_quat = self.get_gripper_pose().quat
            if np.allclose(current_quat, target_quat, atol=1e-3):
                return True

            # Calculate the difference between current and target orientation
            diff_quat = quat_multiply(target_quat, quat_conjugate(current_quat))

            # Convert to axis-angle representation
            axis_angle = quat2axisangle(diff_quat)

            # Limit the maximum rotation per step
            max_rotation = 0.1  # radians
            if np.linalg.norm(axis_angle) > max_rotation:
                axis_angle = axis_angle / np.linalg.norm(axis_angle) * max_rotation

            # Create an action to rotate towards the target orientation
            action = np.zeros(7)  # Assuming 7 DoF robot
            action[3:6] = axis_angle

            self.step(action)

        return False

    def move_to_fingertip_pos(
        self, fingertip_pos: np.ndarray, tar_quat: np.ndarray
    ) -> bool:
        ee_pos = self.fingertip_pos_to_ee(fingertip_pos, tar_quat)
        return self.move_to_pose(Pose(ee_pos, tar_quat))

    def fingertip_pos_to_ee(
        self, fingertip_pos: np.ndarray, ee_quat: np.ndarray
    ) -> np.ndarray:
        home_euler = Rotation.from_quat(self.HOME_QUAT).as_euler("zyx", degrees=True)
        ee_euler = Rotation.from_quat(ee_quat).as_euler("zyx", degrees=True)
        offset_euler = ee_euler - home_euler
        fingertip_offset_euler = offset_euler * [1, -1, 1]
        fingertip_transf = Rotation.from_euler(
            "zyx", fingertip_offset_euler, degrees=True
        )
        fingertip_offset = fingertip_transf.as_matrix() @ self.FINGERTIP_OFFSET
        fingertip_offset[2] -= self.FINGERTIP_OFFSET[2]
        ee_pos = fingertip_pos - fingertip_offset
        return ee_pos

    def ee_pos_to_fingertip(
        self, ee_pos: np.ndarray, ee_quat: np.ndarray
    ) -> np.ndarray:
        current_z_axis = self.extract_z_axis(ee_quat)
        tip_pos = 0.095 * current_z_axis + ee_pos
        return tip_pos

    def extract_z_axis(self, quat: np.ndarray) -> np.ndarray:
        return quat2mat(quat)[:, 2]

    def rotate_around_gripper_z_axis(
        self, angle: float, quat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        angle = np.clip(angle, -89, 89)
        theta_rad = np.radians(angle)
        cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)
        r_gd = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
        current_pose = self.get_gripper_pose()
        if quat is None:
            quat = current_pose.quat
        t_bg = quat2mat(quat)
        r_bg = t_bg[:3, :3]
        r_bd = r_bg @ r_gd
        return mat2quat(r_bd)

    def tilt_updown(self, degrees: float) -> np.ndarray:
        print(f"[DEBUG] Tilting gripper up/down by {degrees} degrees")
        current_quat = self.get_gripper_pose().quat
        print(f"[DEBUG] Current gripper quaternion: {current_quat}")

        # Create a rotation around the Y-axis (which should be the gripper's local X-axis)
        tilt_rotation = Rotation.from_euler("y", np.radians(degrees))

        # Apply this rotation to the current orientation
        current_rotation = Rotation.from_quat(current_quat)
        new_rotation = current_rotation * tilt_rotation

        new_quat = new_rotation.as_quat()
        print(f"[DEBUG] New gripper quaternion after tilt: {new_quat}")
        return new_quat

    def tilt_leftright(self, degrees: float) -> np.ndarray:
        print(f"[DEBUG] Tilting gripper left/right by {degrees} degrees")
        current_quat = self.get_gripper_pose().quat
        print(f"[DEBUG] Current gripper quaternion: {current_quat}")

        # Create a rotation around the Z-axis
        tilt_rotation = Rotation.from_euler("z", np.radians(degrees))

        # Apply this rotation to the current orientation
        current_rotation = Rotation.from_quat(current_quat)
        new_rotation = current_rotation * tilt_rotation

        new_quat = new_rotation.as_quat()
        print(f"[DEBUG] New gripper quaternion after tilt: {new_quat}")
        return new_quat

    def get_horizontal_ori(self) -> np.ndarray:
        print("[DEBUG] Entering get_horizontal_ori")

        # We want the z-axis of the end-effector to be parallel to the ground
        # Let's choose [1, 0, 0] as our target z-axis (pointing along the x-axis)
        target_z = np.array([1.0, 0.0, 0.0])

        # The y-axis should point up
        target_y = np.array([0.0, 0.0, 1.0])

        # Calculate the x-axis to complete the right-handed coordinate system
        target_x = np.cross(target_y, target_z)

        # Construct the rotation matrix
        R = np.column_stack((target_x, target_y, target_z))

        # Convert the rotation matrix to a quaternion
        result = mat2quat(R)

        print(f"[DEBUG] Horizontal orientation quaternion: {result}")
        return result

    def get_vertical_ori(self) -> np.ndarray:
        # We want the z-axis of the end-effector to align with [0, 0, -1]
        target_z = np.array([0.0, 0.0, -1.0])

        # We can choose any perpendicular vector for the y-axis
        # Let's use [0, 1, 0] as it's already perpendicular to our target z-axis
        target_y = np.array([0.0, 1.0, 0.0])

        # Calculate the x-axis to complete the right-handed coordinate system
        target_x = np.cross(target_y, target_z)

        # Construct the rotation matrix
        R = np.column_stack((target_x, target_y, target_z))

        # Convert the rotation matrix to a quaternion
        return mat2quat(R)

    def approach_object(self, object_id: str, distance: float = 0.05) -> bool:
        object_pose = self.get_object_pose(object_id)
        if object_pose is None or object_pose.pos is None:
            print(f"[ERROR] Object {object_id} not found or has invalid position.")
            return False
        gripper_pose = self.get_gripper_pose()
        direction = object_pose.pos - gripper_pose.pos
        direction_norm = direction / np.linalg.norm(direction)
        target_pos = object_pose.pos - direction_norm * distance
        return self.move_to_pose(Pose(target_pos, gripper_pose.quat))

    def lift_object(self, height: float) -> bool:
        current_pose = self.get_gripper_pose()
        target_pos = current_pose.pos.copy()
        target_pos[2] = height
        return self.move_to_pose(Pose(target_pos, current_pose.quat))

    def place_object(self, target_pos: np.ndarray) -> bool:
        current_pose = self.get_gripper_pose()
        if not self.move_to_pose(Pose(target_pos, current_pose.quat)):
            return False
        return self.open_gripper()

    def execute_trajectory(self, waypoints: List[Pose]) -> bool:
        for waypoint in waypoints:
            if not self.move_to_pose(waypoint):
                return False
        return True

    def get_object_in_gripper(self) -> Optional[str]:
        if self.gripper_width > 0.005:
            object_poses = self.get_all_object_poses()
            gripper_pose = self.get_gripper_pose()
            for obj_id, obj_pose in object_poses.items():
                if np.linalg.norm(obj_pose.pos - gripper_pose.pos) < 0.05:
                    return obj_id
        return None

    def reset(self, reset_gripper: bool = True) -> None:
        self.obs = self.env.reset()
        if reset_gripper:
            self.open_gripper()
        self.gripper_width = self.obs["robot0_gripper_qpos"][0]

    def check_grasp(self, object_name):
        object_pos = self.get_object_pose(object_name).pos
        gripper_pos = self.get_gripper_pose().pos
        distance = np.linalg.norm(object_pos - gripper_pos)
        return distance < 0.05  # Adjust this threshold as needed
