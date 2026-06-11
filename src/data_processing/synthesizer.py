import pandas as pd
import json
import os
import random

class DatasetSynthesizer:
    """
    Chapter 3.3: Instruction Dataset Synthesis (SOTA Upgraded).
    Pairs physiological personas with dialogue intents, and generates 
    Proactive JITAI (Just-In-Time Adaptive Intervention) scenarios.
    """
    def __init__(self, vitals_path, dialogues_path):
        self.vitals_df = pd.read_csv(vitals_path)
        self.dialogues_df = pd.read_csv(dialogues_path)
        
    def categorize_dialogue(self, text):
        """Simple intent classifier based on keywords."""
        text = str(text).lower()
        if any(w in text for w in ['panic', 'breath', 'scared', 'heart']):
            return 'anxiety'
        if any(w in text for w in ['hopeless', 'worthless', 'death', 'tired', 'sleep']):
            return 'depression'
        return 'general'

    def generate_jitai_prompt(self, row):
        """Generates a proactive intervention based on individualized Z-score deviations."""
        persona = row['clinical_persona']
        risk_state = row.get('risk_state', 'Stable')
        
        # Only trigger JITAI for elevated risk states
        if risk_state in ['Stable']:
            return None
            
        instruction = f"System: You are an active Bioadaptive Mental Health Agent. Monitor the user's physiological data stream continuously. If thresholds are breached, initiate a Just-In-Time Adaptive Intervention (JITAI) WITHOUT waiting for user input.\n\n{persona}\n\nClient: [SILENT/NO_INPUT]"
        
        # Determine specific JITAI response based on specific Z-score deviations
        if row.get('hrv_z_score', 0) < -2.0: # HRV dropped by more than 2 std dev
            output = "[PROACTIVE JITAI - CRITICAL STRESS] I noticed your Heart Rate Variability has dropped significantly below your normal baseline, indicating high acute stress. Would you like to do a quick 2-minute ACT grounding exercise with me right now?"
        elif row.get('wakeup_z_score', 0) > 2.0: # Wakeups increased by more than 2 std dev
            output = "[PROACTIVE JITAI - SEVERE SLEEP DISRUPTION] Your sleep data shows unusual fragmentation tonight compared to your normal patterns. I am here if you need a calming presence or want to try some progressive muscle relaxation to get back to sleep."
        elif row.get('hr_z_score', 0) > 1.5:
            output = "[PROACTIVE JITAI - ELEVATED HR] Your resting heart rate is noticeably higher than your personal baseline. Let's take a deep breath together. Inhale for 4 seconds, hold for 4, exhale for 6."
        elif risk_state in ['Clinically Concerning', 'Crisis-like']:
            output = f"[PROACTIVE JITAI - {risk_state.upper()} STATE] Your physiological markers are showing concerning deviations from your baseline. I'm here to support you. How are you feeling right now?"
        else:
            return None # No specific threshold crossed despite mildly elevated state
            
        return {"instruction": instruction, "input": "", "output": output}

    def synthesize(self, output_path, limit=2000):
        # Filter personas for better matching
        anxiety_personas = self.vitals_df[self.vitals_df['gad_total'] >= 5]
        depression_personas = self.vitals_df[self.vitals_df['phq_total'] >= 5]
        general_personas = self.vitals_df

        sft_data = []
        
        # 1. Generate Proactive JITAI cases directly from physiology (No user text)
        jitai_count = 0
        for _, row in self.vitals_df.iterrows():
            jitai_case = self.generate_jitai_prompt(row)
            if jitai_case:
                sft_data.append(jitai_case)
                jitai_count += 1
                
        print(f"Generated {jitai_count} proactive JITAI scenarios based purely on bio-markers.")

        # 2. Generate standard reactive dialogue cases (Physio + Text)
        for idx, row in self.dialogues_df.iterrows():
            if len(sft_data) >= limit + jitai_count:
                break
                
            intent = self.categorize_dialogue(row['Context'])
            
            # Match persona based on intent
            if intent == 'anxiety' and not anxiety_personas.empty:
                persona = anxiety_personas.sample(1).iloc[0]['clinical_persona']
            elif intent == 'depression' and not depression_personas.empty:
                persona = depression_personas.sample(1).iloc[0]['clinical_persona']
            else:
                persona = general_personas.sample(1).iloc[0]['clinical_persona']
            
            # Construct Instruction (Physiological Gating simulation)
            instruction = f"System: You are a professional mental health assistant with physiological gating. Modulate your empathy and clinical depth based on the following physiological state.\n\n{persona}\n\nClient: {row['Context']}"
            
            sft_data.append({
                "instruction": instruction,
                "input": "",
                "output": row['Response']
            })

        # Save to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in sft_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        return len(sft_data)

if __name__ == "__main__":
    VITALS_PATH = 'data/processed/deep_vitals_aligned.csv'
    # 使用真实的 80% 训练集数据，而不是完整的全量数据
    DIALOGUES_PATH = 'data/real_splits/kaggle_train.csv'
    OUTPUT_PATH = 'data/processed/sft_train_data.jsonl'
    
    if os.path.exists(VITALS_PATH) and os.path.exists(DIALOGUES_PATH):
        synthesizer = DatasetSynthesizer(VITALS_PATH, DIALOGUES_PATH)
        count = synthesizer.synthesize(OUTPUT_PATH)
        print(f"Successfully synthesized {count} instruction pairs for SFT (including JITAI).")
        print(f"Dataset saved to {OUTPUT_PATH}")
    else:
        print("Required data files not found. Please run the extractor and real_data_splitter first.")
