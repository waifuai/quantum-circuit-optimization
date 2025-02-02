# Quantum Circuit Optimization Problem

Definition of the `tensor2tensor` problem for the task of 
`Generate an quantum circuit response for an quantum circuit input sentence.`.


# Quantum Circuit Optimization with Transformer

This project explores the application of Transformer models to the problem of quantum circuit optimization. 

## Problem Definition

The goal is to train a Transformer model that can learn to generate an optimized quantum circuit given an input quantum circuit. This can be framed as a sequence-to-sequence problem, where the input and output are representations of quantum circuits.

## Approach

We use the `trax` framework to define a `QuantumCircuitOptimizationProblem` and train a Transformer model on it. The problem is defined as a `Text2TextProblem`, where the input and output are treated as sequences of tokens representing quantum gates and operations.

The Transformer model is trained to minimize the difference between the generated output circuit and the target optimized circuit.

## Data

The training data consists of pairs of input and output quantum circuits. The input circuits are randomly generated, and the output circuits are optimized versions of the input circuits.

## Usage

To train the model, you can use `trax`. 

## Evaluation

The model is evaluated on a held-out set of input-output circuit pairs. The evaluation metric is the accuracy of the generated output circuit compared to the target optimized circuit.

## Future Work

* Explore different representations of quantum circuits for input and output.
* Investigate the use of reinforcement learning for training the model.
* Apply the model to real-world quantum circuit optimization problems.