# Run different batch size
for model_name in meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B Qwen/Qwen2.5-3B; do
    for batch_size in 1 2 4 8 16; do
        for sort_dataset in True False; do
            for method in correct; do
                if [ "$sort_dataset" = "True" ]; then
                    python3 reproduce.py \
                        --model_name $model_name \
                        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
                        --batch_size $batch_size \
                        --max_seq_length 2048 \
                        --max_eval_samples 512 \
                        --seed 42 \
                        --method $method \
                        --sort_dataset
                else
                    python3 reproduce.py \
                        --model_name $model_name \
                        --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
                        --batch_size $batch_size \
                        --max_seq_length 2048 \
                        --max_eval_samples 512 \
                        --seed 42 \
                        --method $method
                fi
            done
        done
    done
done