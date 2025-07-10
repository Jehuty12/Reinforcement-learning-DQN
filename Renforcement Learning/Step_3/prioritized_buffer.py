import random
import numpy as np
import torch
from sum_tree import SumTree

class PrioritizedReplayBuffer:
    """
    Buffer de Rejouabilité avec Priorité (Prioritized Experience Replay - PER)
    Cette version est optimisée avec une structure de données SumTree pour
    un échantillonnage et une mise à jour efficaces en temps O(log N).
    """
    def __init__(self, capacity, device="cpu",
                 alpha=0.6, beta_start=0.4, beta_increment=1e-6, eps=1e-6):
        """
        Args:
            capacity (int): Capacité maximale du buffer.
            device (str): Périphérique sur lequel créer les tenseurs ('cpu' ou 'cuda').
            alpha (float): Contrôle le degré de priorité (0=uniforme, 1=priorité directe).
            beta_start (float): Valeur initiale de beta pour l'importance-sampling.
            beta_increment (float): Incrément de beta à chaque échantillonnage.
            eps (float): Petite constante ajoutée aux priorités pour éviter les zéros.
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.eps = eps
        
        # Le stockage des transitions (state, action, reward, next_state, done)
        self.data = [None] * capacity 
        self.pos = 0 # Pointeur pour l'écriture
        self.n_entries = 0 # Nombre d'entrées actuelles

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une nouvelle transition au buffer et au SumTree.
        """
        # 1. Déterminer la priorité pour la nouvelle transition.
        #    On utilise la priorité maximale pour s'assurer qu'elle soit vue rapidement.
        max_prio = self.tree.total() / self.n_entries if self.n_entries > 0 else 1.0
        
        # 2. Stocker la transition dans notre tableau de données.
        transition = (state, action, reward, next_state, done)
        self.data[self.pos] = transition
        
        # 3. Ajouter la transition au SumTree avec sa priorité.
        self.tree.add(max_prio**self.alpha, self.pos)

        # 4. Mettre à jour les pointeurs.
        self.pos = (self.pos + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1


    def sample(self, batch_size):
        """
        Échantillonne un batch de transitions en utilisant le SumTree de manière robuste.
        """
        batch_data = []
        tree_indices = np.empty((batch_size,), dtype=np.int32)
        priorities = np.empty((batch_size,), dtype=np.float32)

        total_p = self.tree.total()
        segment_length = total_p / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        # Utiliser une boucle "while" pour garantir la collecte d'un batch complet
        # d'échantillons valides.
        i = 0
        while i < batch_size:
            s = random.uniform(segment_length * i, segment_length * (i + 1))
            
            tree_idx, prio, data_idx = self.tree.get(s)
            
            # S'assurer que l'index de donnée est valide (ne pointe pas vers un None)
            if data_idx >= self.n_entries:
                # Si l'index est invalide, on ré-échantillonne pour ce segment.
                # On ne fait pas i += 1, on retente la même itération.
                continue

            tree_indices[i] = tree_idx
            priorities[i] = prio
            batch_data.append(self.data[data_idx])
            
            i += 1 # On passe à l'échantillon suivant

        sampling_probs = priorities / total_p

        weights = (self.n_entries * (sampling_probs + 1e-10)) ** -self.beta
        weights /= weights.max() 

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch_data))

        states      = torch.from_numpy(states).float().to(self.device) / 255.0
        next_states = torch.from_numpy(next_states).float().to(self.device) / 255.0
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)
        weights     = torch.from_numpy(weights).float().to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, tree_indices

    def update_priorities(self, tree_indices, td_errors):
        """
        Met à jour la priorité des transitions dans l'arbre après un train_step.
        
        Args:
            tree_indices (np.array): Indices des feuilles dans le SumTree.
            td_errors (np.array): Erreurs temporelles (TD-errors) pour chaque transition.
        """
        new_priorities = np.abs(td_errors) + self.eps
        
        for idx, prio in zip(tree_indices, new_priorities):
            self.tree.update(idx, prio**self.alpha)

    def __len__(self):
        """Retourne le nombre actuel d'éléments dans le buffer."""
        return self.n_entries