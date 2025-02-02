import os
import numpy as np
import trax
from trax.supervised import training
from trax import layers as tl
from src.trainer.problem import get_data_pipelines, get_model

def train_model(input_file: str, output_file: str, model_dir: str,
                batch_size: int = 64, n_steps: int = 1000) -> training.Loop:
    """
    Trains the Transformer model.

    Args:
        input_file: Path to the file containing input circuits.
        output_file: Path to the file containing target circuits.
        model_dir: Directory to save the trained model.
        batch_size: Number of samples per batch.
        n_steps: Number of training steps.

    Returns:
        The training loop object.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Get the data pipelines for training and evaluation
    train_pipeline = get_data_pipelines(input_file, output_file, batch_size)
    eval_pipeline = get_data_pipelines(input_file, output_file, batch_size)

    # Instantiate the model in training mode
    model = get_model(mode='train')

    # Define the training task
    train_task = training.TrainTask(
        labeled_data=train_pipeline,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        n_steps_per_checkpoint=100,
    )

    # Define the evaluation task with appropriate metrics
    eval_task = training.EvalTask(
        labeled_data=eval_pipeline,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )

    # Create and run the training loop
    loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=model_dir,
    )
    loop.run(n_steps=n_steps)

    return loop

def main():
    # Example usage; replace file paths as needed
    input_file = "input.txt"
    output_file = "output.txt"
    model_dir = "model"
    train_model(input_file, output_file, model_dir)

if __name__ == "__main__":
    main()
