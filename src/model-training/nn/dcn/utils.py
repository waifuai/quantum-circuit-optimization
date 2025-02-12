import datetime
import os
import tensorflow as tf
import pandas as pd

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped log and model save paths."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)

def dataframe_to_dataset(dataframe, batch_size, shuffle=True):
    """Convert Pandas DataFrame to tf.data.Dataset."""
    target = ['statevector_00000']
    dataframe = dataframe.copy()
    labels = dataframe.pop(target[0])  # Assuming only one target
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds