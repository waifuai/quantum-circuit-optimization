import datetime
import os

def create_save_paths(logdir: str, modeldir: str, topic: str) -> (str, str):
    """Generate timestamped log and model directories."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subdir = os.path.join(timestamp, topic)
    return os.path.join(logdir, subdir), os.path.join(modeldir, subdir)