import pandas as pd
import os
import glob
import numpy as np

def process_resilient_data(base_path):
    """
    Chapter 3.1 & 3.2: Dataset Profiling and Physio-Semantic Alignment.
    Extracts PHQ/GAD scores and merges with average watch vitals.
    """
    # 1. Load Demographics
    demo_path = os.path.join(base_path, 'Demographics.csv')
    if not os.path.exists(demo_path):
        raise FileNotFoundError(f"Demographics not found at {demo_path}")
        
    demo_df = pd.read_csv(demo_path)
    
    vitals_summary = []
    
    for _, row in demo_df.iterrows():
        uid = row['user_id']
        # Handle potential NaN or missing values
        phq = row.get('phq_total', 0)
        gad = row.get('gad_total', 0)
        
        # Paths to device data (Corrected directory structure)
        user_folder = os.path.join(base_path, 'Sleepmat_Watch_Data', 'Sleepmat_Watch_Data', uid)
        
        user_data = {
            'user_id': uid,
            'phq_total': phq,
            'gad_total': gad,
            'mean_hr': np.nan,
            'mean_steps': np.nan
        }
        
        if os.path.exists(user_folder):
            # Process Heart Rate
            hr_path = os.path.join(user_folder, 'ScanWatch_HR.csv')
            if os.path.exists(hr_path):
                hr_df = pd.read_csv(hr_path)
                if 'Heart Rate' in hr_df.columns:
                    user_data['mean_hr'] = hr_df['Heart Rate'].mean()
            
            # Process Steps
            steps_path = os.path.join(user_folder, 'ScanWatch_Steps.csv')
            if os.path.exists(steps_path):
                steps_df = pd.read_csv(steps_path)
                if 'Steps' in steps_df.columns:
                    user_data['mean_steps'] = steps_df['Steps'].mean()

        vitals_summary.append(user_data)
    
    result_df = pd.DataFrame(vitals_summary)
    
    # 3.2 Physio-Semantic Alignment Logic
    def get_semantic_desc(row):
        """Converts raw vitals into a natural language context string."""
        desc = f"Client context: Mental health scores show PHQ-9 of {row['phq_total']} and GAD-7 of {row['gad_total']}."
        if not np.isnan(row['mean_hr']):
            status = "elevated" if row['mean_hr'] > 85 else "stable"
            desc += f" Observed average heart rate is {row['mean_hr']:.1f} bpm ({status})."
        if not np.isnan(row['mean_steps']):
            desc += f" Physical activity level: {row['mean_steps']:.1f} steps/avg."
        return desc
        
    result_df['vitals_description'] = result_df.apply(get_semantic_desc, axis=1)
    return result_df

if __name__ == "__main__":
    BASE_DATA_PATH = 'references/resilient_data'
    try:
        processed_df = process_resilient_data(BASE_DATA_PATH)
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        processed_df.to_csv('data/processed/vitals_aligned.csv', index=False)
        
        print(f"Successfully processed {len(processed_df)} users.")
        print("\nSample Semantic Alignment:")
        print(processed_df['vitals_description'].iloc[0])
    except Exception as e:
        print(f"Error during processing: {e}")
