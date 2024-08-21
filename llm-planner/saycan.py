import os
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up PyBullet
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


class PickPlaceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.objects = [
            "red block",
            "blue block",
            "green block",
            "yellow block",
            "red bowl",
            "blue bowl",
            "green bowl",
            "yellow bowl",
        ]
        self.reset()

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.objects) * (len(self.objects) - 1))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
        )

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.object_ids = {}
        for obj in self.objects:
            color, shape = obj.split()
            if shape == "block":
                self.object_ids[obj] = p.loadURDF(
                    "cube.urdf", basePosition=self._random_position()
                )
            elif shape == "bowl":
                self.object_ids[obj] = p.loadURDF(
                    "assets/bowl.urdf", basePosition=self._random_position()
                )

            # Set color
            color_rgba = self._color_to_rgba(color)
            p.changeVisualShape(self.object_ids[obj], -1, rgbaColor=color_rgba)

        return self._get_observation()

    def step(self, action):
        obj1, obj2 = self._action_to_objects(action)
        obj1_pos, _ = p.getBasePositionAndOrientation(self.object_ids[obj1])
        obj2_pos, _ = p.getBasePositionAndOrientation(self.object_ids[obj2])

        # Move obj1 to obj2
        p.resetBasePositionAndOrientation(
            self.object_ids[obj1],
            [obj2_pos[0], obj2_pos[1], obj2_pos[2] + 0.1],
            p.getQuaternionFromEuler([0, 0, 0]),
        )

        observation = self._get_observation()
        reward = 1.0  # Simple reward for now
        done = False
        info = {}

        return observation, reward, done, info

    def _random_position(self):
        return [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0.1]

    def _color_to_rgba(self, color):
        color_map = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1],
            "green": [0, 1, 0, 1],
            "yellow": [1, 1, 0, 1],
        }
        return color_map[color]

    def _action_to_objects(self, action):
        obj1_idx = action // (len(self.objects) - 1)
        obj2_idx = action % (len(self.objects) - 1)
        if obj2_idx >= obj1_idx:
            obj2_idx += 1
        return self.objects[obj1_idx], self.objects[obj2_idx]

    def _get_observation(self):
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=128,
            height=128,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = rgba[:, :, :3]
        return rgb

    def render(self):
        img = self._get_observation()
        plt.imshow(img)
        plt.show()


class AffordanceModel(nn.Module):
    def __init__(self, input_channels=3, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.contiguous().view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class LanguageModel:
    def __init__(self):
        self.client = client

    def generate_scores(self, instruction, objects):
        prompt = f"""Instruction: {instruction}
Available objects: {', '.join(objects)}

Rate how relevant each object is to the instruction on a scale of 0 to 1, where 1 is most relevant.
Format your response as a Python dictionary, like this:
{{"object1": score1, "object2": score2, ...}}
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps with robotic task planning.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Parse the response
        try:
            scores = eval(response.choices[0].message.content)
            return scores
        except:
            print("Error parsing GPT-4 response. Returning equal scores.")
            return {obj: 1.0 for obj in objects}


class SayCan:
    def __init__(self, env, language_model, affordance_model):
        self.env = env
        self.language_model = language_model
        self.affordance_model = affordance_model

    def get_action(self, instruction):
        obs = self.env._get_observation()
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        language_scores = self.language_model.generate_scores(
            instruction, self.env.objects
        )
        affordance_scores = self.affordance_model(obs_tensor).squeeze().detach().numpy()

        combined_scores = {}
        for action in range(self.env.action_space.n):
            obj1, obj2 = self.env._action_to_objects(action)
            lang_score = language_scores[obj1] * language_scores[obj2]
            aff_score = affordance_scores[action]
            combined_scores[action] = lang_score * aff_score

        best_action = max(combined_scores, key=combined_scores.get)
        return best_action


class PickPlaceDataset(Dataset):
    def __init__(self, env, num_samples):
        self.env = env
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = self.env.reset()
        action = self.env.action_space.sample()
        return obs, action


def visualize_sample_environments(env, num_samples=5):
    for i in range(num_samples):
        obs = env.reset()
        plt.figure(figsize=(5, 5))
        plt.imshow(obs)
        plt.title(f"Sample Environment {i+1}")
        plt.savefig(f"sample_env_{i+1}.png")
        plt.close()


def visualize_predictions(model, env, num_samples=5):
    for i in range(num_samples):
        obs = env.reset()
        obs_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            predictions = model(obs_tensor).squeeze()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(obs)
        plt.title("Environment")

        plt.subplot(1, 2, 2)
        plt.bar(range(len(predictions)), predictions.numpy())
        plt.title("Action Predictions")
        plt.savefig(f"prediction_sample_{i+1}.png")
        plt.close()


def train_affordance_model(model, dataset, num_epochs, batch_size, learning_rate):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_obs, batch_actions in dataloader:
            optimizer.zero_grad()

            print(f"Batch observation shape: {batch_obs.shape}")
            print(f"Batch actions shape: {batch_actions.shape}")
            print(f"Batch observation min: {batch_obs.min()}, max: {batch_obs.max()}")

            batch_obs = batch_obs.permute(0, 3, 1, 2).float() / 255.0
            print(f"Permuted batch observation shape: {batch_obs.shape}")

            predictions = model(batch_obs)
            print(f"Predictions shape: {predictions.shape}")

            targets = torch.zeros(
                batch_actions.size(0), model.fc2.out_features, device=predictions.device
            )
            targets[torch.arange(batch_actions.size(0)), batch_actions] = 1
            print(f"Targets shape: {targets.shape}")

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")
    plt.close()


# Main execution
if __name__ == "__main__":
    env = PickPlaceEnv()
    language_model = LanguageModel()
    affordance_model = AffordanceModel(input_channels=3, output_dim=env.action_space.n)

    # Visualize sample environments
    visualize_sample_environments(env)

    # Training
    dataset = PickPlaceDataset(env, num_samples=1000)
    train_affordance_model(
        affordance_model, dataset, num_epochs=10, batch_size=32, learning_rate=0.001
    )

    # Visualize predictions
    visualize_predictions(affordance_model, env)

    # SayCan setup
    saycan = SayCan(env, language_model, affordance_model)

    # Demo
    instruction = "Put the red block on the blue bowl"
    print("Instruction:", instruction)
    print("Initial state:")
    env.render()

    action = saycan.get_action(instruction)
    obj1, obj2 = env._action_to_objects(action)
    print(f"Chosen action: Move {obj1} to {obj2}")

    env.step(action)
    print("Final state:")
    env.render()

    plt.show()  # This will display all the plots
