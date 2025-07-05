
python3 reproduce.py \
    --model_name meta-llama/Llama-3.2-1B \
    --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
    --batch_size 32 \
    --max_seq_length 2048 \
    --max_eval_samples 512 \
    --seed 42 \
    --sort_dataset \
    --method correct
