# Neurips2025_Cautious_Optimizers
Anonymous release for cautious optimizers

### Pretraining Llama on C4
```bash
torchrun --standalone --nproc_per_node 1 torchrun_main.py \
         --model_config configs/llama_60m.json \
         --lr 0.001 \
         --batch_size 16 \
         --total_batch_size 512 \
         --activation_checkpointing \
         --num_training_steps 10000 \
         --warmup_steps 1000 \
         --weight_decay 0 \
         --grad_clipping 1.0 \
         --dtype bfloat16 \
         --eval_every 1000 \
         --single_gpu \ # remove this if multi gpus
         --optimizer c-adamw \
         --max_length 256
```
Rebuttal run on 100M for up to 50B tokens
```bash
torchrun --rdzv_id=$SLURM_JOB_ID \
         --rdzv-backend=c10d \
         --nnodes=16 \
         --nproc_per_node 1 \
         torchrun_main.py \
         --model_config configs/llama_100m.json \
         --lr 0.001 \
         --batch_size 16 \
         --total_batch_size 2048 \
         --activation_checkpointing \
         --num_training_steps 25000 \
         --warmup_steps 1000 \
         --weight_decay 0.1 \
         --grad_clipping 1.0 \
         --dtype bfloat16 \
         --eval_every 1000 \
         --optimizer c-adamw \ # replace with adamw for adamw run
         --betas 0.9 0.95 \
         --max_length 1024
```

---

### Pretraining MAE on ImageNet 1K (50 Epochs)
```bash
torchrun --standalone --nproc_per_node 4 run_mae.py \
    --dataset_name ILSVRC/imagenet-1k \
    --output_dir ./vit-mae-c \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 50 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --custom_optim c-adamw \
    --trust_remote_code \
    --gradient_accumulation_steps 4
```
### Post Training Qwen2.5
```
torchrun \
    --rdzv_id=$JOB_ID \
    --rdzv-backend=c10d \
    --nnodes=1:8 \
    --nproc-per-node=1 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    post_training.py --model "Qwen/Qwen2.5-1.5B-Instruct" \
                     --output_dir cautious_1.5b \
                     --per_device_train_batch_size 1 \
                     --gradient_accumulation_steps 2 \
                     --max_length 8192 \
                     --cautious
```

---

### PPO
```
accelerate launch ppo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \     
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100 \
    --custom_optim c_adamw \
    --num_gpus 8
```
