import tensorflow as tf

def decode_csv(example):
    """Decodes a CSV line into features and labels."""
    # Define the structure of the CSV record
    record_defaults = [tf.float32] * 32 + [tf.string] * 41  # 32 labels + 41 gate operation strings
    
    # Decode the CSV record
    decoded_record = tf.io.decode_csv(example, record_defaults=record_defaults)
    
    # Separate features and labels
    labels = tf.stack(decoded_record[:32])
    gate_operations = decoded_record[32:]
    return gate_operations, labels

def parse_gate_operation(gate_string):
    """Parses a gate operation string into a GateOperationData object."""
    parts = tf.strings.split(gate_string, sep='|')
    gate_type = parts[0]
    control = tf.strings.to_number(parts[1], out_type=tf.float32)
    target = tf.strings.to_number(parts[2], out_type=tf.float32)
    angle1 = tf.strings.to_number(parts[3], out_type=tf.float32)
    angle2 = tf.strings.to_number(parts[4], out_type=tf.float32)
    angle3 = tf.strings.to_number(parts[5], out_type=tf.float32)
    
    return gate_type, control, target, angle1, angle2, angle3

def preprocess_data(gate_operations, labels):
    """Preprocess gate operations and labels with optional data augmentation."""
    
    def augment_gate(gate_type, control, target, angle1, angle2, angle3):
        """Applies data augmentation to a single gate operation."""
        from ds_dnn import config
        if config.DATA_AUGMENTATION:
            # Add small random rotations to gate angles
            angle1 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)
            angle2 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)
            angle3 += tf.random.uniform(shape=[], minval=-0.1, maxval=0.1, dtype=tf.float32)

            # Swap control and target qubits in CNOT gates
            swap_prob = 0.1  # Probability of swapping control and target
            if tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32) < swap_prob:
                control, target = target, control
        return gate_type, control, target, angle1, angle2, angle3

    processed_gates = []
    for gate_string in gate_operations:
        gate_type, control, target, angle1, angle2, angle3 = parse_gate_operation(gate_string)
        gate_type, control, target, angle1, angle2, angle3 = augment_gate(gate_type, control, target, angle1, angle2, angle3)
        processed_gates.append((gate_type, control, target, angle1, angle2, angle3))

    return processed_gates, labels