import numpy as np


def weighted_random_sample(items: np.array, weights: np.array, n: int) -> np.array:
    """
    Does np.random.choice but ensuring we don't have duplicates in the final result

    Args:
        items (np.array): _description_
        weights (np.array): _description_
        n (int): _description_

    Returns:
        np.array: _description_
    """
    indices = np.arange(len(items))
    out_indices = []

    for _ in range(n):
        chosen_index = np.random.choice(indices, p=weights)
        out_indices.append(chosen_index)

        mask = indices != chosen_index
        indices = indices[mask]
        weights = weights[mask]

        if weights.sum() != 0:
            weights = weights / weights.sum()

    return items[out_indices]
