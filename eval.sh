
# Run different batch size
for batch_size in 1 2 4 8 16; do
    python3 reproduce.py \
        --model_name meta-llama/Llama-3.2-1B \
        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
        --batch_size $batch_size \
        --max_seq_length 2048 \
        --max_eval_samples 2048 \
        --seed 42
done