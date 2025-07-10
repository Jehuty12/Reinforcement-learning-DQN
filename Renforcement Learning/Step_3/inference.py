import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, AtariPreprocessing
from preprocessing import FrameStack
from model import DQN

def run_inference(
    env_id="ALE/Breakout-v5",
    model_path="dqn_breakout.pth",
    video_folder="inference_videos",
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Préparation de l'environnement avec enregistrement vidéo
    base_env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
        screen_size=84
    )
    os.makedirs(video_folder, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,  # Enregistre le premier épisode
        name_prefix="inference"
    )

    # Chargement du modèle
    obs, _ = env.reset()
    frame_stack = FrameStack(4, device=device)
    state = frame_stack.reset(obs)
    n_actions = env.action_space.n
    policy_net = DQN(input_channels=4, num_actions=n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            q_values = policy_net(state)
            action = q_values.argmax().item()  # Exploitation pure (pas d'exploration)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = frame_stack.step(obs)
        total_reward += reward

    env.close()
    print(f"Épisode terminé. Score total : {total_reward}")

if __name__ == "__main__":
    run_inference()