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
    return np.mean(data[-window_size:]) if len(data) >= window_size else np.mean(data)

def sum_of_last_100(data):
    return np.sum(data[-100:]) if len(data) >= 100 else np.sum(data)

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
    im = ax.imshow(np.zeros(grid_shape), cmap='Blues', origin='upper', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Q-value")
    ax.set_title("Learned Q-values\nArrows represent best action")
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    return im, cbar

def update_q_table_plot(Q, im, ax, grid_shape=(4, 12)):
    for txt in list(ax.texts):
        txt.remove()
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    max_q = np.max(Q, axis=1).reshape(grid_shape)
    im.set_data(max_q)
    vmin, vmax = np.min(max_q), np.max(max_q)
    im.set_clim(vmin, vmax)
    for s in range(Q.shape[0]):
        r, c = divmod(s, grid_shape[1])
        best = np.argmax(Q[s])
        color = 'white' if max_q[r, c] > (vmin + vmax) / 2 else 'black'
        ax.text(c, r, arrows[best], ha='center', va='center', color=color)

# --------------------------------------------------
# SARSA (On-policy) principal
# --------------------------------------------------
def sarsa_cliffwalking_dashboard():
    episodes = 1500  # augmenter le nombre d'épisodes pour meilleure convergence
    max_steps = 200
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.0  # laisser epsilon descendre à 0 pour une politique greedy pure
    decay_rate = 0.995  # facteur de décroissance exponentielle par épisode

    # initialiser epsilon et paramètres pour décroissance multiplicative
    # epsilon_start = 1.0
    # epsilon_min = 0.0
    # epsilon = epsilon_start

    env = gym.make("CliffWalking-v0")
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rewards, lengths, eps_list = [], [], []
    sum_rewards, avg_actions = [], []

    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1])
    axEnv = fig.add_subplot(gs[0, 0])
    axQ = fig.add_subplot(gs[0, 1:3])
    axR = fig.add_subplot(gs[1, 0])
    axA = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[1, 2])
    im_q, _ = init_q_table_plot(axQ)

    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        # action initiale on-policy epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        total_reward = 0

        for t in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            # Choix de next_action on-policy
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            # Mise à jour SARSA
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            if done or truncated:
                break

        # Stats
        rewards.append(total_reward)
        lengths.append(t + 1)
        eps_list.append(epsilon)
        sum_rewards.append(sum_of_last_100(rewards))
        avg_actions.append(moving_average(lengths, 100))
        epsilon = max(epsilon_min, epsilon * decay_rate)  # décroissance multiplicative de epsilon

        if ep % 10 == 0:
            render_cliffwalking(state, axEnv)
            update_q_table_plot(Q, im_q, axQ)
            axR.clear(); axR.plot(sum_rewards); axR.set_title("Sum last 100 rewards")
            axA.clear(); axA.plot(avg_actions); axA.set_title("Moving avg actions")
            axE.clear(); axE.plot(eps_list); axE.set_title("Epsilon decay")
            plt.pause(0.001)

    plt.ioff(); plt.show(); env.close()
    return Q, rewards, lengths, eps_list

# --------------------------------------------------
# Simulation visuelle en mode human
# --------------------------------------------------
def simulate_policy_human(Q, delay=0.1):
    env = gym.make("CliffWalking-v0", render_mode="human")
    state, _ = env.reset()
    env.render(); total_reward = 0
    while True:
        action = np.argmax(Q[state])
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        env.render(); time.sleep(delay)
        if done or truncated:
            break
    print(f"Terminé: reward={total_reward}")
    env.close()

if __name__ == '__main__':
    Q, rewards, lengths, eps_vals = sarsa_cliffwalking_dashboard()
    simulate_policy_human(Q)
