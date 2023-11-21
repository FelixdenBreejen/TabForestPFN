import numpy as np


def create_random_sequences(
    default_value_val: float, 
    default_value_test: float,
    random_values_val: np.ndarray, 
    random_values_test: np.ndarray,
    sequence_length: int,
    n_shuffles: int
):
    """
    Makes random test sequences.
    Let random_values_val and random_values_test be arrays of shape (n_runs,), which are the scores of the sweep.
    We are interested what happens if we would have executed this sweep in a different order.
    We pick sequence_length random values from random_values_val and random_values_test, randomize the order (with replacement), and prepend the default values.
    We track the running-best validation score, and return the matching test score for each sequence.
    The number of sequences is n_shuffles.

    returns:
        best_test_score: np.ndarray of shape (n_shuffles, sequence_length)
    """

    
    assert len(random_values_val) == len(random_values_test), "The number of random values for val and test must be the same"

    if len(random_values_val) == 0:
        # We consider default runs (no random values) as a drawn horizontal line
        return np.tile(default_value_test, (n_shuffles, sequence_length))

    random_values = np.concatenate([random_values_val[None, :], random_values_test[None, :]], axis=0)
    default_values = np.array([default_value_val, default_value_test])

    random_index = np.random.randint(0, len(random_values_val), size=(n_shuffles, sequence_length-1))

    random_sequences = random_values[:, random_index]
    sequences = np.concatenate([np.tile(default_values[:, None], (1, n_shuffles))[:, :, None], random_sequences], axis=2)

    best_validation_score = np.maximum.accumulate(sequences[0, :, :], axis=1)
    diff = best_validation_score[:, :-1] < best_validation_score[:, 1:]
    diff_prepend_zeros = np.concatenate([np.zeros((n_shuffles, 1), dtype=bool), diff], axis=1)
    best_validation_idcs = diff_prepend_zeros * np.arange(sequence_length)[None, :]
    best_validation_idcs = np.maximum.accumulate(best_validation_idcs, axis=1)

    best_test_score = sequences[1, np.arange(n_shuffles)[:, None], best_validation_idcs ]
    
    return best_test_score
    


if __name__ == '__main__':

    seq = create_random_sequences(5, 6, np.array([3, 8, 3, 6, 3, 3, 7, 4, 3]), np.array([4, 5, 3, 7, 3, 4, 7, 3, 4]), 5, 3)
    print(seq)
    pass

