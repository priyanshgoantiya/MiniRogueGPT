BabyGPT

A miniature autoregressive GPT-style language model implemented in PyTorch.

BabyGPT is a compact transformer-based language model for educational and experimental purposes. It can train on custom text datasets, generate text, and demonstrate the fundamentals of GPT-style autoregressive language modeling.

Features

Autoregressive Text Generation – Generates text token by token.

Transformer Architecture – Multi-head self-attention with feedforward layers and GELU activation.

Causal Masking – Prevents the model from looking ahead during training.

Configurable Hyperparameters – Easily adjust embedding size, number of layers, attention heads, and more.

Text Generation – Generate sequences of arbitrary length from a seed context.

Checkpoint Saving – Save model checkpoints during training for future use.

Requirements
Python 3.8+
PyTorch 2.0+
NumPy

