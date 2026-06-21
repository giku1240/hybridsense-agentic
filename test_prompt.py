import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = 'Qwen/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(model, 'models/hybrid-sense-dora_full')

prompt = """System: You are an active Bioadaptive Mental Health Agent. Monitor the user's physiological data stream continuously. If thresholds are breached, initiate a Just-In-Time Adaptive Intervention (JITAI) WITHOUT waiting for user input.

Psychometric Profile: PHQ-9=3, GAD-7=3. Overall Risk State: Mildly Elevated. Physiological Deviation: HR is elevated at 72.8bpm (Baseline: 58.0, Z-score: 1.67); HRV(SDNN) has stable to 64.1ms (Baseline: 60.5, Z-score: 0.16); Sleep wakeups stable to 7 (Baseline: 54.3).

Client: [SILENT/NO_INPUT]"""

inputs = tokenizer(prompt + "\n\n", return_tensors='pt').to('cuda')
out = model.generate(**inputs, max_new_tokens=50)
print('--- FULL OUTPUT ---')
print(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
