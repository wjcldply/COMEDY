# COMEDY
This is the Re-Proudction project of paper: [**Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations**](https://arxiv.org/abs/2402.11975)

## Ko-COMEDY DataSet
- Need to save `task1_dataset/`, `task1_dataset/`, and `task1_dataset/` directories under `Data/ko_COMEDY/` direcotry.
- Run the following bash commands to create Dataset for Fine-Tuning.
   ```bash
   cd Data/ko_COMEDY
   python load_ko_COMEDY.py
   ```

## LoRA FineTuning
- Run following command to check whether fine-tuning + saving works as intended (with 100 sample data)
   ```bash
   python training/step1_supervised_finetuning/main_peft.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --test \
      --context_window 4096 \
      --lora_rank 32 \
      --epochs 3 \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --checkpointing_ratio 0.25 \
      --fp16 \
      --wandb_run_name test
   ```
- Remove any logs or checkpoint models from test run (above), and run the command below to conduct actual fine-tuning
   ```bash
   rm -rf Models/*
   rm -rf wandb/*
   ```
   ```bash
   python training/step1_supervised_finetuning/main_peft.py \
      --model_path meta-llama/Llama-3.1-8B-Instruct \
      --context_window 4096 \
      --lora_rank 32 \
      --epochs 3 \
      --per_device_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --checkpointing_ratio 0.25 \
      --fp16 \
      --wandb_run_name lora_finetuning_run_1
   ```

   ```
   nohup python training/step1_supervised_finetuning/main_peft.py \
      --model_path meta-llama/Llama-3.2-1B-Instruct \
      --context_window 4096 \
      --lora_rank 64 \
      --epochs 3 \
      --per_device_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --checkpointing_ratio 0.1 \
      --wandb_run_name llama3.2-1B-lora-finetuning > finetuning_output.log 2>&1 &
   ```