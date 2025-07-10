import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt

# Paramètres de base
base_E0 = 12000
base_tau = 3000
base_E0_max = 18000
epsilon_max = 1.0

# Fonctions d'ajustement
def compute_map_difficulty(desc):
    num_traps = np.sum(desc == 'H')
    total = desc.size
    return num_traps / total

def adjust_parameters(base_E0, base_tau, base_E0_max, difficulty, avg_reward):
    performance_factor = 1 + (1 - avg_reward)  # entre 1 et 2
    E0_adj = base_E0 * (1 + difficulty) * performance_factor
    tau_adj = base_tau * (1 + difficulty) * performance_factor
    E0_max_adj = base_E0_max * (1 + difficulty) * performance_factor
    return E0_adj, tau_adj, E0_max_adj

def compute_epsilon(episode, dynamic_E0, tau):
    return epsilon_max / (1 + np.exp((episode - dynamic_E0) / tau))

def plot_q_table(ax, Q, grid_size=8, trap_states=None):
    ax.clear()
    ax.set_title("Q-table (annotée)")
    max_q = np.max(Q)
    normalized_Q = Q / max_q if max_q > 0 else np.zeros_like(Q)
    best_actions = np.argmax(Q, axis=1)
    max_q_values = np.max(normalized_Q, axis=1).reshape((grid_size, grid_size))
    if np.max(max_q_values) == 0:
        max_q_values += 0.01  
    heatmap = ax.imshow(max_q_values, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    directions = ['←', '↓', '→', '↑']
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            if trap_states is not None and state in trap_states:
                continue
            action = best_actions[state]
            ax.text(j, i, directions[action], ha='center', va='center', color='black', fontsize=12)
    return heatmap

def run():
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.delaxes(axes[0, 2])
    ax_frame = axes[0, 0]
    ax_qtable = axes[0, 1]
    ax_rewards = axes[1, 0]
    ax_steps = axes[1, 1]
    ax_epsilon = axes[1, 2]

    line_rewards, = ax_rewards.plot([], [], label="Rolling rewards (100 eps)")
    ax_rewards.set_title("Somme des rewards (fenêtre 100)")
    ax_rewards.set_xlabel("Épisode")
    ax_rewards.set_ylabel("Somme des rewards")
    ax_rewards.legend()

    ax_qtable.set_title("Q-table (annotée)")
    line_epsilon, = ax_epsilon.plot([], [], label="Epsilon", color="green")
    ax_epsilon.set_title("Évolution d'epsilon")
    ax_epsilon.set_xlabel("Épisode")
    ax_epsilon.set_ylabel("Epsilon")
    ax_epsilon.legend()

    line_steps, = ax_steps.plot([], [], label="Rolling avg steps (100 eps)", color="orange")
    ax_steps.set_title("Moyenne du nombre d'actions (fenêtre 100)")
    ax_steps.set_xlabel("Épisode")
    ax_steps.set_ylabel("Nombre d'actions moyen")
    ax_steps.legend()

    # Création d'une map aléatoire et extraction de la description
    random_map = generate_random_map(size=8)
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=False,
                   render_mode="rgb_array", desc=random_map)
    desc = env.unwrapped.desc.astype('U1')
    difficulty = compute_map_difficulty(desc)
    print(f"Map difficulty: {difficulty:.2f}")

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    grid_size = 8
    trap_states = [i * grid_size + j for i in range(grid_size)
                   for j in range(grid_size) if desc[i, j] == 'H']

    alpha = 0.1
    gamma = 0.9
    max_steps = 200

    # Phase d'exploration initiale pour 500 épisodes
    initial_episodes = 500
    rewards_initial = []
    for ep in range(initial_episodes):
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_steps = 0
        while not (terminated or truncated) and episode_steps < max_steps:
            action = env.action_space.sample()  # exploration totale
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            episode_steps += 1
            # Ajustement de la récompense comme dans votre code
            if terminated and reward == 1:
                reward = 1 - (episode_steps / max_steps)
            else:
                reward = 0
            episode_reward += reward
        rewards_initial.append(episode_reward)
    avg_reward_initial = np.mean(rewards_initial)
    print(f"Average reward over first {initial_episodes} episodes: {avg_reward_initial:.2f}")

    # Ajustement des paramètres dynamiques en fonction de la difficulté et des performances initiales
    dynamic_E0, tau, dynamic_E0_max = adjust_parameters(base_E0, base_tau, base_E0_max, difficulty, avg_reward_initial)
    print(f"Adjusted parameters: dynamic_E0 = {dynamic_E0:.0f}, tau = {tau:.0f}, dynamic_E0_max = {dynamic_E0_max:.0f}")

    num_episodes = 40000
    rewards_per_episode = []
    rolling_sums = []
    eps_values = []
    steps_per_episode = []
    rolling_steps = []
    success_per_episode = []

    window_size = 100
    update_interval = 500
    last_frame = None
    cbar = None
    episode = 0
    while episode < num_episodes:
        state, _ = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        episode_steps = 0

        current_epsilon = compute_epsilon(episode, dynamic_E0, tau)
        while not (terminated or truncated) and episode_steps < max_steps:
            if np.random.rand() < current_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                last_frame = env.render()
            if terminated and reward == 1:
                reward = 1 - (episode_steps / max_steps)
            else:
                reward = 0
            episode_reward += reward
            episode_steps += 1
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            state = next_state

        rewards_per_episode.append(episode_reward)
        steps_per_episode.append(episode_steps)
        success_per_episode.append(int(episode_reward > 0))
        rolling_sum = np.sum(rewards_per_episode[-window_size:]) if len(rewards_per_episode) >= window_size else np.sum(rewards_per_episode)
        rolling_sums.append(rolling_sum)
        avg_steps = np.mean(steps_per_episode[-window_size:]) if len(steps_per_episode) >= window_size else np.mean(steps_per_episode)
        rolling_steps.append(avg_steps)
        eps_values.append(current_epsilon)

        # Mécanisme d'ajustement dynamique (tel que vous avez déjà)
        if episode_reward == 0:
            if dynamic_E0 < dynamic_E0_max:
                dynamic_E0 += 200
        else:
            dynamic_E0 = max(base_E0, dynamic_E0 - 100)
        if episode_reward == 0:
            num_episodes += 1

        # Early stopping sur une fenêtre de 1000 épisodes si l'agent réussit systématiquement
        if len(success_per_episode) >= 1000:
            success_rate = np.mean(success_per_episode[-1000:])
            if success_rate >= 1.0:
                print(f"Early stopping at episode {episode}: success rate = {success_rate:.2f}")
                break

        if episode % update_interval == 0 and episode > 0:
            line_rewards.set_xdata(np.arange(len(rolling_sums)))
            line_rewards.set_ydata(rolling_sums)
            ax_rewards.set_xlim(0, len(rolling_sums))
            ax_rewards.set_ylim(0, 100)
            heatmap = plot_q_table(ax_qtable, Q, grid_size=grid_size, trap_states=trap_states)
            if cbar is None:
                cbar = plt.colorbar(heatmap, ax=ax_qtable, fraction=0.046, pad=0.04)
                cbar.set_ticks(np.linspace(0, 1, 5))
                cbar.set_label("Pertinence", rotation=270, labelpad=15)
            else:
                cbar.update_normal(heatmap)
            line_epsilon.set_xdata(np.arange(len(eps_values)))
            line_epsilon.set_ydata(eps_values)
            ax_epsilon.set_xlim(0, len(eps_values))
            ax_epsilon.set_ylim(0, 1.05)
            line_steps.set_xdata(np.arange(len(rolling_steps)))
            line_steps.set_ydata(rolling_steps)
            ax_steps.set_xlim(0, len(rolling_steps))
            ax_steps.set_ylim(10, 40)
            if last_frame is not None:
                ax_frame.clear()
                ax_frame.imshow(last_frame)
                ax_frame.set_title("Dernière frame")
            plt.draw()
            plt.pause(0.01)
        episode += 1

    env.close()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run()
