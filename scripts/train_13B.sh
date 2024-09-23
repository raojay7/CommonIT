deepspeed --master_addr "localhost" \
    --master_port 22346 \
   train_fast.py \
    --deepspeed deepspeed_zero2.conf \
    --model_name_or_path "/data/llama-13b" \
    --model_max_length 2048 \
    --data_path_dir alpaca_len \
    --output_dir checkpoints_13b_deep/ \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 1 \
  --learning_rate 2.5e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "linear" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --report_to wandb


