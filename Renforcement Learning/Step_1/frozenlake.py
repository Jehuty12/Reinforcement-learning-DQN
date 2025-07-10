import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import gridspec


def exponential_decay(epsilon_initial, epsilon_min, decay_rate, episode):
    return epsilon_min + (epsilon_initial - epsilon_min) * np.exp(-decay_rate * episode)


def shape_reward(original_reward, done, success_value=1.0):
    step_penalty = -0.01
    if done and original_reward == success_value:
        return original_reward + 5.0 + step_penalty
    elif done and original_reward == 0:
        return original_reward - 2.0 + step_penalty
    else:
        return original_reward + step_penalty


def get_last_game_frame(Q, env):
    """
    Simule un épisode en suivant la politique greedy (selon Q)
    et retourne la dernière frame (image RGB) du jeu.
    """
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    frame = env.render()
    return frame


def update_all_plots(ax_game, ax_epsilon, ax_q, ax_actions, ax_rewards,
                     game_im, heatmap_im, heatmap_cbar,
                     epsilon_history, rewards_per_episode, actions_per_episode,
                     Q, env_frame, window_size):
    # Calcule la taille de la grille à partir de la description de l'environnement
    desc = env_frame.unwrapped.desc
    grid_size = desc.shape[0] if hasattr(desc, 'shape') else len(desc)

    # Mise à jour de la dernière frame du jeu
    frame = get_last_game_frame(Q, env_frame)
    game_im.set_data(frame)
    ax_game.set_title("Dernière frame du jeu")
    ax_game.axis("off")

    # Mise à jour de l'évolution d'epsilon
    ax_epsilon.clear()
    ax_epsilon.plot(epsilon_history, label="Epsilon", color="tab:blue")
    ax_epsilon.set_xlabel("Épisode")
    ax_epsilon.set_ylabel("Valeur d'Epsilon")
    ax_epsilon.set_title("Évolution d'Epsilon")
    ax_epsilon.legend()
    ax_epsilon.grid(True)

    # Préparation de l'affichage de la Q-table
    arrow_map = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    best_Q = np.zeros((grid_size, grid_size))
    best_action_grid = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            best_action = np.argmax(Q[state])
            best_action_grid[i, j] = best_action
            best_Q[i, j] = Q[state, best_action]

    # Création d'un masque pour les cases d'eau (marquées par 'H' ou b'H')
    water_mask = np.array([[True if (cell == b'H' or cell == "H") else False for cell in row] for row in desc])
    best_Q[water_mask] = np.nan  # Affichage des cases d'eau en blanc

    # Mise à jour de l'image via imshow
    heatmap_im.set_data(best_Q)
    heatmap_cbar.update_normal(heatmap_im)

    # Actualisation des annotations (pas de flèches sur l'eau)
    for txt in ax_q.texts:
        txt.remove()
    for i in range(grid_size):
        for j in range(grid_size):
            if water_mask[i, j]:
                continue
            # Récupère la Q-valeur pour la cellule et détermine la couleur de fond via le colormap
            q_val = best_Q[i, j]
            if np.isnan(q_val):
                continue
            # Conversion de la Q-valeur en couleur (RGBA) via la normalisation et le colormap utilisé
            rgba = heatmap_im.cmap(heatmap_im.norm(q_val))
            # Calcul de la luminosité perçue (formule de luminance)
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            # Choix de la couleur du texte selon un seuil (par exemple 0.5)
            arrow_color = "white" if brightness < 0.5 else "black"
            ax_q.text(j, i, arrow_map[best_action_grid[i, j]],
                      ha="center", va="center", color=arrow_color, fontsize=8)
    ax_q.set_title("Q‑table (meilleure action)")

    # Mise à jour de la moyenne glissante du nombre d'actions
    ax_actions.clear()
    if len(actions_per_episode) >= window_size:
        cumsum = np.cumsum(np.insert(actions_per_episode, 0, 0))
        rolling_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        x_values = np.arange(window_size, len(actions_per_episode) + 1)
        ax_actions.plot(x_values, rolling_avg, label=f"Moyenne (fenêtre={window_size})", color="tab:green")
        ax_actions.set_xlabel("Épisode")
        ax_actions.set_ylabel("Nombre d'actions")
        ax_actions.set_title("Moyenne glissante des actions")
        ax_actions.legend()
        ax_actions.grid(True)

    # Mise à jour de la somme glissante des récompenses
    ax_rewards.clear()
    if len(rewards_per_episode) >= window_size:
        cumsum = np.cumsum(np.insert(rewards_per_episode, 0, 0))
        rolling_sum = cumsum[window_size:] - cumsum[:-window_size]
        x_values = np.arange(window_size, len(rewards_per_episode) + 1)
        ax_rewards.plot(x_values, rolling_sum, label=f"Somme (fenêtre={window_size})", color="tab:red")
        ax_rewards.set_xlabel("Épisode")
        ax_rewards.set_ylabel("Somme des récompenses")
        ax_rewards.set_title("Somme glissante des récompenses")
        ax_rewards.legend()
        ax_rewards.grid(True)


def train_agent(env, episodes=30000, max_steps=100, alpha=0.1, gamma=0.9,
                epsilon=1.0, epsilon_min=0.01, decay_rate=0.0005, window_size=100,
                update_plots=False, update_interval=100, env_frame=None):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    epsilon_history = []
    rewards_per_episode = []
    actions_per_episode = []
    success_history = []
    initial_epsilon = epsilon

    # Détermine grid_size à partir de l'environnement (ou via racine carrée du nombre d'états)
    if env_frame is not None:
        desc = env_frame.unwrapped.desc
        grid_size = desc.shape[0] if hasattr(desc, 'shape') else len(desc)
    else:
        grid_size = int(np.sqrt(num_states))

    if update_plots:
        plt.ion()
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        ax_game = fig.add_subplot(gs[0, 0])
        ax_epsilon = fig.add_subplot(gs[0, 1])
        ax_q = fig.add_subplot(gs[0, 2])
        ax_actions = fig.add_subplot(gs[1, 0:2])
        ax_rewards = fig.add_subplot(gs[1, 2])
        fig.tight_layout(pad=3.0)

        # Configuration du colormap pour la heatmap (NaN en blanc)
        cmap = cm.get_cmap("Blues").copy()
        cmap.set_bad(color="white")

        # Affichage initial de la dernière frame du jeu
        env_frame.reset()
        initial_frame = env_frame.render()
        game_im = ax_game.imshow(initial_frame)
        ax_game.set_title("Dernière frame du jeu")
        ax_game.axis("off")

        # Affichage initial de la Q‑table
        initial_best_Q = np.zeros((grid_size, grid_size))
        heatmap_im = ax_q.imshow(initial_best_Q, cmap=cmap, origin="upper")
        heatmap_cbar = fig.colorbar(heatmap_im, ax=ax_q, fraction=0.046, pad=0.04)
    else:
        game_im = None
        heatmap_im = None
        heatmap_cbar = None
        ax_game = ax_epsilon = ax_q = ax_actions = ax_rewards = None

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0

        for _ in range(max_steps):
            step_count += 1

            # Choix de l'action à faire en fonction du taux d'epsilon
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            shaped_reward = shape_reward(reward, done, success_value=1.0)

            episode_reward += shaped_reward

            best_next_action = np.argmax(Q[next_state])
            td_target = shaped_reward + gamma * Q[next_state, best_next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            if done:
                break

        rewards_per_episode.append(episode_reward)
        actions_per_episode.append(step_count)
        success = 1 if episode_reward > 0 else 0
        success_history.append(success)
        epsilon = exponential_decay(initial_epsilon, epsilon_min, decay_rate, episode)
        epsilon_history.append(epsilon)

        if update_plots and episode % update_interval == 0:
            update_all_plots(ax_game, ax_epsilon, ax_q, ax_actions, ax_rewards,
                             game_im, heatmap_im, heatmap_cbar,
                             epsilon_history, rewards_per_episode, actions_per_episode,
                             Q, env_frame, window_size)
            plt.pause(0.001)

    print("Entraînement terminé !")
    if update_plots:
        plt.ioff()
        plt.show()

    return Q, epsilon_history, rewards_per_episode, actions_per_episode


def simulate_agent(env, Q):
    state, _ = env.reset()
    env.metadata['render_fps'] = 10
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    print("Simulation terminée.")


def save_final_display(Q, env, filename="final_display.png"):
    """
    Crée et enregistre une figure finale affichant la dernière frame du jeu
    et la Q‑table avec annotations (flèches indiquant la meilleure action).
    """
    final_frame = env.render()
    desc = env.unwrapped.desc
    grid_size = desc.shape[0] if hasattr(desc, 'shape') else len(desc)

    def qtable_directions_map(qtable, grid_size):
        qtable_val_max = qtable.max(axis=1).reshape(grid_size, grid_size)
        qtable_best_action = np.argmax(qtable, axis=1).reshape(grid_size, grid_size)
        directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
        eps = np.finfo(float).eps
        for idx, val in enumerate(qtable_best_action.flatten()):
            if qtable_val_max.flatten()[idx] > eps:
                qtable_directions[idx] = directions[val]
            else:
                qtable_directions[idx] = ""
        qtable_directions = qtable_directions.reshape(grid_size, grid_size)
        return qtable_val_max, qtable_directions

    qtable_val_max, qtable_directions = qtable_directions_map(Q, grid_size)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(final_frame)
    ax[0].axis("off")
    ax[0].set_title("Dernière frame du jeu")

    import seaborn as sns
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    )
    ax[1].set_title("Learned Q-values\nArrows represent best action")

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Final display saved as {filename}")


def main():
    # Définir la taille souhaitée de la carte
    map_size = 4  # Par exemple, pour une carte 10x10

    # Pour les tailles 4x4 et 8x8, on peut utiliser le nom de carte prédéfini.
    # Pour toute autre taille, on génère une carte personnalisée.
    allowed_maps = ["4x4", "8x8"]
    map_name = f"{map_size}x{map_size}"
    if map_name not in allowed_maps:
        generated_map = generate_random_map(size=map_size, p=0.8)
        map_name = None  # On utilise uniquement le paramètre `desc`
    else:
        generated_map = None

    # Environnements d'entraînement et pour affichage (avec rendu en rgb_array)
    env_train = gym.make("FrozenLake-v1", map_name=map_name, render_mode=None, is_slippery=False, desc=generated_map)
    env_frame = gym.make("FrozenLake-v1", map_name=map_name, render_mode="rgb_array", is_slippery=False,
                         desc=generated_map)

    episodes = 1000
    max_steps = 200
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    decay_rate = 0.00005
    window_size = 100

    Q, epsilon_history, rewards_per_episode, actions_per_episode = train_agent(
        env_train, episodes, max_steps, alpha, gamma, epsilon, epsilon_min, decay_rate,
        window_size=window_size,
        update_plots=True,
        update_interval=200,
        env_frame=env_frame
    )
    env_train.close()
    env_frame.close()

    # Environnement pour la simulation avec rendu humain
    env_sim = gym.make("FrozenLake-v1", map_name=map_name, render_mode="human", is_slippery=False, desc=generated_map)
    simulate_agent(env_sim, Q)
    save_final_display(Q, env_sim)
    env_sim.close()


if __name__ == "__main__":
    main()
