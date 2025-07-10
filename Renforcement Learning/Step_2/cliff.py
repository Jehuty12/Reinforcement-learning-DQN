import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import time

# --------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------
def moving_average(data, window_size=100):
    if len(data) < window_size:
        return np.mean(data)
    return np.mean(data[-window_size:])

def sum_of_last_100(data):
    if len(data) < 100:
        return np.sum(data)
    return np.sum(data[-100:])

# --------------------------------------------------
# Rendu du plateau
# --------------------------------------------------
def render_cliffwalking(state, ax, grid_shape=(4, 12)):
    ax.clear()
    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.set_xticks(np.arange(0, grid_shape[1] + 1, 1))
    ax.set_yticks(np.arange(0, grid_shape[0] + 1, 1))
    ax.grid(True)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            color = 'white'
            if i == grid_shape[0] - 1:
                if j == 0:
                    color = 'green'
                elif j == grid_shape[1] - 1:
                    color = 'blue'
                else:
                    color = 'red'
            ax.add_patch(patches.Rectangle((j, grid_shape[0] - 1 - i), 1, 1,
                                          facecolor=color, edgecolor='black'))
    r, c = divmod(state, grid_shape[1])
    ax.add_patch(patches.Circle((c + 0.5, grid_shape[0] - 1 - r + 0.5), 0.3, color='black'))
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Last frame")

# --------------------------------------------------
# Q-table plotting
# --------------------------------------------------
def init_q_table_plot(ax, grid_shape=(4, 12)):
    data_init = np.zeros(grid_shape)
    im = ax.imshow(data_init, cmap='Blues', origin='upper', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Q-value")
    ax.set_title("Learned Q-values\nArrows represent best action")
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    ax.set_aspect('equal')
    return im, cbar

def update_q_table_plot(Q, im, ax, grid_shape=(4, 12)):
    for txt in list(ax.texts):
        txt.remove()
    arrow_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    max_q_values = np.max(Q, axis=1).reshape(grid_shape)
    vmax_val = max(1.0, np.max(Q))
    im.set_data(max_q_values)
    im.set_clim(0, vmax_val)
    for s in range(Q.shape[0]):
        row, col = divmod(s, grid_shape[1])
        best_action = np.argmax(Q[s])
        arrow = arrow_symbols[best_action]
        val = max_q_values[row, col]
        color = 'white' if val > (vmax_val / 2.0) else 'black'
        ax.text(col, row, arrow, ha='center', va='center', color=color, fontsize=12)

# --------------------------------------------------
# Q-learning principal (off-policy)
# --------------------------------------------------
def q_learning_cliffwalking_dashboard():
    # Paramètres
    episodes = 1500
    max_steps = 200
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    decay_rate = 0.0005

    env = gym.make("CliffWalking-v0")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    episode_lengths = []
    eps_list = []
    sum_rewards_list = []
    avg_actions_list = []
    episodes_list = []

    # Figure interactive
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1])
    axEnv = fig.add_subplot(gs[0, 0])
    axQ = fig.add_subplot(gs[0, 1:3])
    axR = fig.add_subplot(gs[1, 0])
    axA = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])
    im_q, _ = init_q_table_plot(axQ)

    for episode in range(episodes):
        state, _ = env.reset(seed=episode)
        total_reward = 0
        step_count = 0

        for _ in range(max_steps):
            # ε-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            best_next = np.max(Q[next_state])
            # Q-learning update
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state
            if terminated or truncated:
                break

        # Stats
        rewards_per_episode.append(total_reward)
        episode_lengths.append(step_count)
        eps_list.append(epsilon)
        sum_rewards_list.append(sum_of_last_100(rewards_per_episode))
        avg_actions_list.append(moving_average(episode_lengths, 100))
        episodes_list.append(episode)
        epsilon = max(epsilon_min, epsilon - decay_rate)

        # Update plots
        if episode % 10 == 0:
            render_cliffwalking(state, axEnv)
            update_q_table_plot(Q, im_q, axQ)
            axR.clear(); axR.plot(episodes_list, sum_rewards_list); axR.set_title("Sum last 100 rewards")
            axA.clear(); axA.plot(episodes_list, avg_actions_list); axA.set_title("Moving avg actions")
            axE.clear(); axE.plot(episodes_list, eps_list); axE.set_title("Epsilon decay")
            plt.pause(0.001)

    plt.ioff(); plt.show(); env.close()
    return Q, rewards_per_episode, episode_lengths, eps_list

# --------------------------------------------------
# Simulation visuelle (mode human)
# --------------------------------------------------
def simulate_policy_human(Q, max_steps=200, delay=0.1):
    """
    Exécute la politique greedy apprise et affiche le rendu humain de l'env.
    Affiche chaque étape et imprime le nombre de pas effectués et la récompense totale.
    """
    env = gym.make("CliffWalking-v0", render_mode="human")
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0

    # Boucle de simulation
    while not done and step < max_steps:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        done = terminated or truncated

        # Affichage et pause
        env.render()
        time.sleep(delay)

        step += 1

    print(f"Simulation terminée en {step} pas, reward totale = {total_reward}")
    env.close()

# --------------------------------------------------
# Lancement du script
# --------------------------------------------------
if __name__ == '__main__':
    Q, rewards, lengths, eps_values = q_learning_cliffwalking_dashboard()
    simulate_policy_human(Q)
