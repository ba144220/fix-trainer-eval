import argparse
import os
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from trl.trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset

from utils import save_results

def vanilla(model, tokenizer, dataset, args):
    sft_config = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )
    eval_result = trainer.evaluate()
    return eval_result

def sft_with_compute_loss(model, tokenizer, dataset, args):
    sft_config = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    def compute_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"].float()
        
        # Shift label
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, logits.size(-1)) # (batch_size * seq_length, vocab_size)
        shift_labels = shift_labels.view(-1) # (batch_size * seq_length)
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")
        return loss

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        compute_loss_func=compute_loss,
    )
    eval_result = trainer.evaluate()
    return eval_result

def correct(model, tokenizer, dataset, args):
    sft_config = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    def compute_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"].float()
        batch_size = logits.shape[0]
        
        # Shift label
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        logits = logits.view(-1, logits.size(-1)) # (batch_size * seq_length, vocab_size)
        shift_labels = shift_labels.view(-1) # (batch_size * seq_length)
        loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="none") # (batch_size * seq_length)
        
        # Reshape loss to (batch_size, seq_length)
        loss = loss.view(batch_size, -1)
        shift_labels = shift_labels.view(batch_size, -1) # (batch_size, seq_length)

        # Sum the loss over the sequence length
        loss = loss.sum(dim=1) # (batch_size)
        
        # Normalize the loss by the number of items in each sequence
        num_items_in_each_sequence = (shift_labels != -100).sum(dim=1) # (batch_size)
        loss = loss / num_items_in_each_sequence # (batch_size)

        loss = loss.mean()

        return loss

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        compute_loss_func=compute_loss,
    )
    eval_result = trainer.evaluate()
    return eval_result

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset_name", type=str, default="togethercomputer/RedPajama-Data-1T-Sample")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_eval_samples", type=int, default=2048)
    parser.add_argument("--sort_dataset", action='store_true', help="Sort dataset by length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_csv", type=str, default="./results.csv")
    parser.add_argument("--method", type=str, default="vanilla")
    args = parser.parse_args()

    print(f"{args=}")

    # model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto", token=os.getenv("HF_TOKEN"), torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset is not a Dataset object")
    dataset = dataset.shuffle(args.seed).select(range(args.max_eval_samples))
    if args.sort_dataset:
        dataset = dataset.map(lambda x: {"length": len(x["text"])}, batched=False, num_proc=16)
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns("length")

    # def preprocess_function(examples):
    #     # texts = [text + tokenizer.eos_token for text in examples["text"]]
    #     texts = examples["text"]
    #     inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    #     return inputs

    # dataset = dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE, num_proc=16)

    if args.method == "vanilla":
        eval_result = vanilla(model, tokenizer, dataset, args)
    elif args.method == "sft_with_compute_loss":
        eval_result = sft_with_compute_loss(model, tokenizer, dataset, args)
    elif args.method == "correct":
        eval_result = correct(model, tokenizer, dataset, args)
    else:
        raise ValueError(f"Method {args.method} not supported")

    print(eval_result)
    save_results(eval_result, args)

if __name__ == "__main__":
    main()