import os
import numpy as np

def load_data(data_dir: str, input_file: str = "input_processed.txt", output_file: str = "output_processed.txt"):
    """
    Loads and preprocesses the quantum circuit data.
    Returns a tuple of (inputs, targets, vocab).
    """
    input_path = os.path.join(data_dir, input_file)
    target_path = os.path.join(data_dir, output_file)
    inputs, targets, vocab = [], [], set()

    with open(input_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            inputs.append(stripped_line)
            vocab.update(list(stripped_line))

    with open(target_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            targets.append(stripped_line)
            vocab.update(list(stripped_line))

    return inputs, targets, sorted(list(vocab))

def data_generator(inputs: list, targets: list, vocab: list, batch_size: int = 16, max_length: int = 256):
    """
    Generator function for training data.
    Yields tuples of (input_batch, target_batch, mask_batch).
    """
    char_to_index = {char: index for index, char in enumerate(vocab)}
    num_batches = len(inputs) // batch_size

    while True:
        for i in range(num_batches):
            input_batch, target_batch = [], []
            for j in range(batch_size):
                index = i * batch_size + j
                input_seq = [char_to_index.get(char, 0) for char in inputs[index]]
                target_seq = [char_to_index.get(char, 0) for char in targets[index]]
                input_seq = input_seq + [0] * (max_length - len(input_seq))
                target_seq = target_seq + [0] * (max_length - len(target_seq))
                input_batch.append(input_seq)
                target_batch.append(target_seq)
            mask_batch = [[1 if token != 0 else 0 for token in seq] for seq in input_batch]
            yield (np.array(input_batch), np.array(target_batch), np.array(mask_batch))
