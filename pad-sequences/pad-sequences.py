import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    n_samples = len(seqs)
    if n_samples == 0:
        return np.zeros((0,0))

    actural_max_len = max(len(seq) for seq in seqs) or 0
    l_target = actural_max_len if max_len is None else max_len

    result = np.full((n_samples, l_target), pad_value, dtype=np.int32)
    
    for i, seq in enumerate(seqs):
        if len(seq) == 0:
            continue 

        data_to_copy = seq[:l_target]
        result[i, :len(data_to_copy)] = data_to_copy

    return result