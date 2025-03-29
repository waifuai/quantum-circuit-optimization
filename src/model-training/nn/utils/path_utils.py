# src/model-training/nn/utils/path_utils.py
import datetime
import os
import config # Assuming config might be needed later

def create_save_paths(logdir_base: str = config.LOG_DIR, 
                      modeldir_base: str = config.MODEL_DIR, 
                      topic: str = "default_topic") -> tuple[str, str]:
    """
    Generate timestamped log and model save paths based on base directories and a topic.

    Args:
        logdir_base: The base directory for logs. Defaults to config.LOG_DIR.
        modeldir_base: The base directory for models. Defaults to config.MODEL_DIR.
        topic: A string identifier for the specific training run or model type.

    Returns:
        A tuple containing the full log directory path and model directory path.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Use os.path.join for cross-platform compatibility
    subdir = os.path.join(timestamp, topic) 
    log_path = os.path.join(logdir_base, subdir)
    model_path = os.path.join(modeldir_base, subdir)
    
    # Ensure the directories exist
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    return log_path, model_path

# Alias for compatibility with norm_ds_dnn usage if needed, or refactor norm_ds_dnn later
def setup_logging_and_model_dirs(logdir_base: str = config.LOG_DIR, 
                                 modeldir_base: str = config.MODEL_DIR, 
                                 topic: str = "cpu_norm_ds_dnn") -> tuple[str, str]:
    """Alias for create_save_paths for backward compatibility or specific use cases."""
    # Note: The original setup_logging_and_model_dirs hardcoded the topic.
    # We might want to pass the topic from the calling script instead.
    # For now, keeping the default topic used in norm_ds_dnn.
    return create_save_paths(logdir_base, modeldir_base, topic)