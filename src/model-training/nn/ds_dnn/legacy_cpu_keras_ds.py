import datetime
import glob
import os
import tensorflow as tf
import re

# Configuration parameters
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 2**15  # ~32k
EPOCHS = int(1e9)
VALIDATION_SPLIT = 0.1

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped directories for logs and models."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

def preprocess_data(example: tf.Tensor):
    """
    Preprocess a single CSV line into features and labels.
    Uses vectorized operations and regex replacement.
    """
    adder = [0.0] * 32 + [1.0, 0.0] * 41 + [1.0] * (41 * 5)
    divisor = [1024.0] * 32 + [float(i + 1), 1.0] * 41 + [5.0] * (41 * 2) + [3.0] * (41 * 3)
    adder_tensor = tf.constant(adder, dtype=tf.float32)
    divisor_tensor = tf.constant(divisor, dtype=tf.float32)

    # Mapping dictionary for replacements
    replace_dict = {
        "U3Gate": "0",
        "CnotGate": "0.333",
        "Measure": "0.667",
        "BLANK": "1",
    }
    
    # Use tf.strings.regex_replace with a lambda wrapper for each key
    for key, value in replace_dict.items():
        example = tf.strings.regex_replace(example, key, value)
    
    numeric_data = tf.strings.to_number(tf.strings.split(example, ","), out_type=tf.float32)
    processed_data = (numeric_data + adder_tensor) / divisor_tensor
    features = processed_data[32:]
    labels = processed_data[:32]
    return features, labels

def create_dataset(file_paths):
    """Create a TensorFlow dataset from CSV file paths."""
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.interleave(
        lambda x: tf.data.TextLineDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.skip(1).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False).batch(BATCH_SIZE)
    return dataset

def main():
    DIR_PATH = "shards/"
    file_paths = sorted(glob.glob(os.path.join(DIR_PATH, "*.csv")))
    
    n_train_files = int((1 - VALIDATION_SPLIT) * len(file_paths))
    train_file_paths = file_paths[:n_train_files]
    val_file_paths = file_paths[n_train_files:]
    
    print(f"Training files ({len(train_file_paths)}): {train_file_paths[0]} -> {train_file_paths[-1]}")
    print(f"Validation files ({len(val_file_paths)}): {val_file_paths[0]} -> {val_file_paths[-1]}")
    
    TOPIC = "cpu_keras_ds_qc9"
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOGDIR = "logs/fit/"
    MODELDIR = "models/"
    
    save_path = os.path.join(TIMESTAMP, TOPIC)
    log_dir, model_dir = create_save_paths(LOGDIR, MODELDIR, TOPIC)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    print(f"Saving best model to: {model_dir}")
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    
    train_dataset = create_dataset(train_file_paths)
    val_dataset = create_dataset(val_file_paths)
    
    # Load an existing model or define a new one if desired:
    MODEL_PATH = "models/qc_dnn_8m_32s"  # Update as needed
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Could not load model from {MODEL_PATH}: {e}")
        # Uncomment and customize the following lines to define a new model:
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(1024, activation='relu', input_shape=(246,)),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dense(256, activation='relu'),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(32)
        # ])
        # model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        exit()
    
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, model_checkpoint],
    )

if __name__ == "__main__":
    main()
