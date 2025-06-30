

# Preprocess dataset

The following two methods are equivalent.

## Method 1
```python
...
def preprocess_function(examples):
    # texts = [text + tokenizer.eos_token for text in examples["text"]]
    texts = examples["text"]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    return inputs

dataset = dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE, num_proc=16)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    eval_dataset=dataset,
)
...
```

## Method 2
```python
# No preprocess dataset

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    eval_dataset=dataset,
    processing_class=tokenizer,
)
```
The SFTTrainer will preprocess the dataset using the tokenizer and add the `EOS` token to the end of the input.
