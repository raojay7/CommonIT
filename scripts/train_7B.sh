torchrun --nproc_per_node 4 --nnodes 1 \
    --master_port 22345 \
   train.py \
    --model_name_or_path "decapoda-research/llama-7b-hf" \
    --model_max_length 2048 \
    --data_path_dir gpt4 \
    --output_dir checkpoints/mytest \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "linear" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --gradient_checkpointing True

