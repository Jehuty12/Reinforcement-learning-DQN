import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0

    def add(self, priority, data_index):
        tree_index = self.data_pointer + self.capacity - 1
        self.update(tree_index, priority)

        # Avancer dans l'arbre (circulaire)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagation du changement vers la racine
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, s):
        """
        Retourne (index de l'arbre, priorité, index de l'échantillon)
        pour une somme cumulative s dans [0, total())
        """
        index = 0
        while True:
            left = 2 * index + 1
            right = left + 1

            if left >= len(self.tree):  # feuille atteinte
                leaf_index = index
                break

            if s <= self.tree[left]:
                index = left
            else:
                s -= self.tree[left]
                index = right

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index

    def total(self):
        return self.tree[0]
