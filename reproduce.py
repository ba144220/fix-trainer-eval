import argparse
import os
import torch
from dotenv import load_dotenv

load_dotenv()



from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from trl.trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset

import pandas as pd

def save_results(eval_result, args):
    args_dict = vars(args)
    if os.path.exists(args.results_csv):
        df = pd.read_csv(args.results_csv)
    else:
        columns = [str(k) for k in args_dict.keys()] + [str(k) for k in eval_result.keys()]
        df = pd.DataFrame(columns=pd.Index(columns))
        
    df = pd.concat([df, pd.DataFrame([{**args_dict, **eval_result}])], ignore_index=True)
    df.to_csv(args.results_csv, index=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--dataset_name", type=str, default="togethercomputer/RedPajama-Data-1T-Sample")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_eval_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_csv", type=str, default="./results.csv")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, token=os.getenv("HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")
    if not isinstance(dataset, Dataset):
        raise ValueError("Dataset is not a Dataset object")
    dataset = dataset.shuffle(args.seed).select(range(args.max_eval_samples))

    # def preprocess_function(examples):
    #     # texts = [text + tokenizer.eos_token for text in examples["text"]]
    #     texts = examples["text"]
    #     inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    #     return inputs

    # dataset = dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE, num_proc=16)

    sft_config = SFTConfig(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataset_text_field="text",       
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
    )


    eval_result = trainer.evaluate()
    print(eval_result)
    save_results(eval_result, args)

if __name__ == "__main__":
    main()