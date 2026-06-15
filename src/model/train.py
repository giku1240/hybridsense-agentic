import os
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

def train(config_dict=None, ablation_mode="full"):
    """
    Chapter 4: DoRA Training with Ablation Modes.
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
        instruction_ids = tokenizer(instruction, truncation=True, max_length=1024, padding=False)["input_ids"]
        labels = [-100] * len(instruction_ids) + tokenized_full["input_ids"][len(instruction_ids):]
        
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
    trainer.train()
    
    # Save the final LoRA weights
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SUCCESS: Model saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "no_hrv", "text_only"])
    args = parser.parse_args()
    
    train(ablation_mode=args.ablation)
