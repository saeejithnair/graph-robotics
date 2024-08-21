import numpy as np
import robosuite as suite
import cv2

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

width, height = env.camera_width, env.camera_height
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('robosuite_video.mp4', fourcc, 30, (width, height))

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

    if 'agentview_image' in obs:
        frame = obs['agentview_image']
    else:
        frame = env.render(mode='rgb_array', camera_name="agentview")
    
    # OpenCV expects BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Write frame to video
    video.write(frame)

# Release video writer
video.release()

print("Video saved as robosuite_video.mp4")
