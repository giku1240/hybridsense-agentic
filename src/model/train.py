import os
import glob
import torch
import yaml
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

def find_latest_checkpoint(output_dir):
    """
    Scan output_dir for checkpoint-* subdirectories and return the one
    with the highest step number, or None if no checkpoints exist.
    """
    pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    # Sort by the numeric step suffix
    checkpoints = sorted(
        checkpoints,
        key=lambda p: int(os.path.basename(p).split("-")[-1])
    )
    return checkpoints[-1]


def train(config_dict=None, ablation_mode="full", resume_from_checkpoint=None):
    """
    Chapter 4: DoRA Training with Ablation Modes.
    
    Args:
        config_dict: Optional config dict. If None, loads from configs/train_config.yaml.
        ablation_mode: One of "full", "no_hrv", "text_only".
        resume_from_checkpoint: Path to a specific checkpoint directory to resume from,
            or True to auto-detect the latest checkpoint in output_dir, or None/False
            to start fresh.
    """
    if config_dict is None:
        # Fallback for direct execution
        with open("configs/train_config.yaml", "r") as f:
            config = yaml.safe_load(f)
    else:
        config = config_dict

    # Modify output dir based on ablation mode
    output_dir = f"{config['output_dir']}_{ablation_mode}"
    print(f"[{ablation_mode.upper()} MODE] Initializing Training Pipeline...")
    print(f"Target Directory: {output_dir}")
    
    # Resolve checkpoint to resume from
    if resume_from_checkpoint is True:
        # Auto-detect latest checkpoint in output_dir
        resume_from_checkpoint = find_latest_checkpoint(output_dir)
        if resume_from_checkpoint:
            print(f"[RESUME] Auto-detected checkpoint: {resume_from_checkpoint}")
        else:
            print("[RESUME] No existing checkpoint found — starting fresh.")
            resume_from_checkpoint = None
    elif resume_from_checkpoint:
        # Explicit path provided
        if not os.path.isdir(resume_from_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {resume_from_checkpoint}"
            )
        print(f"[RESUME] Resuming from specified checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    print(f"Loading dataset from: {config['dataset_path']}")
    dataset = load_dataset("json", data_files=config['dataset_path'], split="train")

    def tokenize_function(examples):
        instruction = examples["instruction"]
        
        # Apply Ablation Filtering
        if ablation_mode == "no_hrv":
            instruction = instruction.split("HRV(SDNN)")[0] if "HRV(SDNN)" in instruction else instruction
        elif ablation_mode == "text_only":
            # Remove the whole Physiological Markers section
            instruction = instruction.split("Physiological Markers:")[0] if "Physiological Markers:" in instruction else instruction

        full_text = f"{instruction}\n\n{examples['output']}{tokenizer.eos_token}"
        tokenized_full = tokenizer(full_text, truncation=True, max_length=1024, padding=False)
        
        # Mask the instruction part
        prompt_text = f"{instruction}\n\n"
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=1024, padding=False)["input_ids"]
        
        num_prompt_tokens = len(prompt_ids)
        if num_prompt_tokens >= len(tokenized_full["input_ids"]):
            labels = [-100] * len(tokenized_full["input_ids"])
        else:
            labels = [-100] * num_prompt_tokens + tokenized_full["input_ids"][num_prompt_tokens:]
            
        tokenized_full["labels"] = labels
        return tokenized_full

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    print(f"Dataset tokenized. Sample size: {len(tokenized_dataset)}")

    # 4. Quantization Config
    if config.get('load_in_4bit') or config.get('load_in_8bit'):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.get('load_in_4bit', False),
            load_in_8bit=config.get('load_in_8bit', False),
            bnb_4bit_compute_dtype=torch.float16 if config.get('bnb_4bit_compute_dtype') == "float16" else torch.float32,
            bnb_4bit_quant_type=config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=config.get('bnb_4bit_use_double_quant', True)
        )
    else:
        bnb_config = None

    # 5. Load Model
    print(f"Loading Base Model: {config['model_name_or_path']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name_or_path'],
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if config.get('fp16') else torch.bfloat16 if config.get('bf16') else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # 6. DoRA Config
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config['use_dora']
    )
    model = get_peft_model(model, peft_config)
    print("DoRA Adapters integrated.")

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        num_train_epochs=config['num_train_epochs'],
        max_steps=config.get('max_steps', -1),
        lr_scheduler_type=config['lr_scheduler_type'],
        warmup_ratio=config['warmup_ratio'],
        fp16=config['fp16'],
        logging_steps=config['logging_steps'],
        save_strategy=config['save_strategy'],
        report_to="wandb",
        remove_unused_columns=False
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    print("Trainer ready. Starting optimization...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final LoRA weights
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SUCCESS: Model saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "no_hrv", "text_only"])
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a specific checkpoint directory to resume from, "
            "or 'auto' to automatically find the latest checkpoint in the output directory."
        )
    )
    args = parser.parse_args()
    # Normalize the --resume_from_checkpoint value
    resume = args.resume_from_checkpoint
    if resume is not None and resume.lower() == "auto":
        resume = True   # Signal find_latest_checkpoint() to auto-detect

    train(ablation_mode=args.ablation)
