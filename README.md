# Fix HF Trainer Evaluation Loss

# Problem

When using the `Trainer` class from the `transformers` library, the evaluation loss will decrease when the batch size is increased.
