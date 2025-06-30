
# Run different batch size
# meta-llama/Llama-3.2-1B, Qwen/Qwen2.5-3B, meta-llama/Llama-3.2-3B
for model_name in meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B; do
    for batch_size in 1; do
        python3 reproduce.py \
            --model_name $model_name \
            --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
            --batch_size $batch_size \
            --max_seq_length 2048 \
            --max_eval_samples 2048 \
            --seed 42 \
            --sort_dataset True
    done
done