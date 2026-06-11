import pandas as pd
import os
import numpy as np

class DeepPhysioExtractor:
    """
    Chapter 3: Individualized Physiological Risk Modeling (SOTA Upgraded).
    Calculates per-participant longitudinal baselines to determine dynamic risk states 
    (R(participant, t)) rather than relying on absolute static thresholds.
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.demo_df = pd.read_csv(os.path.join(base_path, 'Demographics.csv'))

    def get_individualized_vitals(self, uid):
        user_folder = os.path.join(self.base_path, 'Sleepmat_Watch_Data', 'Sleepmat_Watch_Data', uid)
        if not os.path.exists(user_folder):
            return None

        features = {}
        
        # 1. Heart Rate & HRV: Longitudinal Baseline Modeling
        physio_path = os.path.join(user_folder, 'Sleep_physio.csv')
        if os.path.exists(physio_path):
            df = pd.read_csv(physio_path)
            
            # Calculate Individual Baselines (Mean and Std Dev over full history)
            hr_baseline = df['Heart Rate'].mean()
            hr_std = df['Heart Rate'].std()
            
            # Simulated 'current day' reading (using the most recent valid block)
            current_hr = df['Heart Rate'].iloc[-50:].mean() # Last 50 readings proxy
            
            # HRV Baseline
            hrv_series = df['SDNN_1'].replace(0, np.nan).dropna()
            if not hrv_series.empty:
                hrv_baseline = hrv_series.mean()
                hrv_std = hrv_series.std()
                current_hrv = hrv_series.iloc[-50:].mean()
            else:
                hrv_baseline, hrv_std, current_hrv = np.nan, np.nan, np.nan
            
            # Determine Risk State (Z-Score deviation from personal baseline)
            # z_score = (current - baseline) / std
            hr_z_score = (current_hr - hr_baseline) / hr_std if hr_std > 0 else 0
            hrv_z_score = (current_hrv - hrv_baseline) / hrv_std if hrv_std > 0 else 0
            
            features.update({
                'hr_baseline': hr_baseline,
                'current_hr': current_hr,
                'hr_z_score': hr_z_score,
                'hrv_baseline': hrv_baseline,
                'current_hrv': current_hrv,
                'hrv_z_score': hrv_z_score
            })

        # 2. Sleep Architecture: Longitudinal Baseline Modeling
        state_path = os.path.join(user_folder, 'Sleep_state.csv')
        if os.path.exists(state_path):
            state_df = pd.read_csv(state_path)
            state_df['Start time'] = pd.to_datetime(state_df['Start time']).dt.date
            
            # Calculate daily wakeups across all historical nights
            daily_wakeups = state_df[state_df['Sleep state'] == 'wakeup'].groupby('Start time').size()
            
            if not daily_wakeups.empty:
                wakeup_baseline = daily_wakeups.mean()
                wakeup_std = daily_wakeups.std()
                current_wakeups = daily_wakeups.iloc[-1] # Most recent night
                
                features.update({
                    'wakeup_baseline': wakeup_baseline,
                    'current_wakeups': current_wakeups,
                    'wakeup_z_score': (current_wakeups - wakeup_baseline) / wakeup_std if wakeup_std > 0 else 0
                })

        return features
      
    def run(self):
        all_data = []
        for _, row in self.demo_df.iterrows():
            uid = row['user_id']
            vitals = self.get_individualized_vitals(uid)
            
            combined = {
                'user_id': uid,
                'phq_total': row.get('phq_total', 0),
                'gad_total': row.get('gad_total', 0)
            }
            if vitals:
                combined.update(vitals)
                all_data.append(combined)
            
        return pd.DataFrame(all_data)

    def determine_risk_state(self, row):
        """
        Maps Z-scores to 4 continuous risk states.
        State: Stable -> Mildly Elevated -> Clinically Concerning -> Crisis-like
        """
        max_z_stress = max(
            row.get('hr_z_score', 0), 
            -row.get('hrv_z_score', 0), # Negative Z-score for HRV means higher stress
            row.get('wakeup_z_score', 0)
        )
        
        if max_z_stress > 3.0:
            return "Crisis-like"
        elif max_z_stress > 2.0:
            return "Clinically Concerning"
        elif max_z_stress > 1.0:
            return "Mildly Elevated"
        else:
            return "Stable"

    def generate_clinical_persona(self, row):
        """Creates a narrative based on individualized baselines."""
        state = self.determine_risk_state(row)
        persona = f"Psychometric Profile: PHQ-9={row['phq_total']}, GAD-7={row['gad_total']}. Overall Risk State: {state}."
        
        physio_parts = []
        if not np.isnan(row.get('hr_z_score', np.nan)):
            direction = "elevated" if row['hr_z_score'] > 0 else "stable"
            physio_parts.append(f"HR is {direction} at {row['current_hr']:.1f}bpm (Baseline: {row['hr_baseline']:.1f}, Z-score: {row['hr_z_score']:.2f})")
        
        if not np.isnan(row.get('hrv_z_score', np.nan)):
            direction = "dropped" if row['hrv_z_score'] < 0 else "stable"
            physio_parts.append(f"HRV(SDNN) has {direction} to {row['current_hrv']:.1f}ms (Baseline: {row['hrv_baseline']:.1f}, Z-score: {row['hrv_z_score']:.2f})")
            
        if not np.isnan(row.get('wakeup_z_score', np.nan)):
            direction = "increased" if row['wakeup_z_score'] > 0 else "stable"
            physio_parts.append(f"Sleep wakeups {direction} to {row['current_wakeups']} (Baseline: {row['wakeup_baseline']:.1f})")

        if physio_parts:
            persona += " Physiological Deviation: " + "; ".join(physio_parts) + "."
        
        return persona

if __name__ == "__main__":
    extractor = DeepPhysioExtractor('references/resilient_data')
    df = extractor.run()
    df['risk_state'] = df.apply(extractor.determine_risk_state, axis=1)
    df['clinical_persona'] = df.apply(extractor.generate_clinical_persona, axis=1)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/deep_vitals_aligned.csv', index=False)
    print(f"Computed individualized baselines for {len(df)} users.")
    print(f"Risk State Distribution:\n{df['risk_state'].value_counts()}")
    print("\nSample Individualized Persona:")
    print(df['clinical_persona'].iloc[0])
