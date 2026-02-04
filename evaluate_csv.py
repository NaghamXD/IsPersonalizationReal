import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
from VSViG import VSViG_base

# --- 1. SETUP & IMPORTS ---
# Ensure paths are correct as fixed previously
BASE_PATH = "/Users/naghamdaood/Projects/VSViG"
METADATA_EXCEL_PATH = "/Users/naghamdaood/Projects/VSViG/WU-SAHZU-EMU-Video/dataset/Label.xlsx"
MASTER_LABEL_FILE = os.path.join(BASE_PATH, "processed_data/labels.json")
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, "processed_data")
MODEL_PATH = os.path.join(BASE_PATH, "checkpoints_improved/best_model.pth")
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class vsvig_dataset(Dataset):
    def __init__(self, data_folder, label_file):
        with open(label_file, 'r') as f:
            self._labels_dict = json.load(f)
        self._keys = list(self._labels_dict.keys())
        self._folder = data_folder

    def __getitem__(self, idx):
        file_key = self._keys[idx]
        label = float(self._labels_dict[file_key])
        
        # Parse: "pat01_Sz1_100" -> UniqueID: "pat01_Sz1", Frame: 100
        parts = file_key.split('_')
        unique_id = f"{parts[0]}_{parts[1]}"
        try: frame_num = int(parts[-1])
        except: frame_num = 0
        
        # Robust path handling
        data_path = os.path.join(self._folder, 'patches', f"{file_key}.pt")
        kpts_path = os.path.join(self._folder, 'kpts', f"{file_key}.pt")
        
        # Fallback for flat directory structure
        if not os.path.exists(data_path):
            data_path = os.path.join(self._folder, f"{file_key}.pt")
            kpts_path = os.path.join(self._folder, f"{file_key}_kpts.pt")

        data = torch.load(data_path, map_location='cpu')
        kpts = torch.load(kpts_path, map_location='cpu')
                # 2. Standardization (Matches your request)
        if data.max() > 2.0: 
            data = data.float() / 255.0
        else:
            data = data.float()
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        data = (data - mean) / std
        
        # --- KEYPOINT NORMALIZATION FIX ---
        kpts = kpts.float()
        kpts[:, :, 0] /= 1920.0
        kpts[:, :, 1] /= 1080.0
        
        return data, kpts, label, unique_id, frame_num

    def __len__(self):
        return len(self._keys)
    
# --- 2. SETTINGS ---
DT_SENSITIVITY = 0.3
STRIDE = 1.0
WINDOW_SIZE = 3
PAPER_WINDOW = -120 # Seconds (Detections earlier than this are False Alarms)

# --- 3. HELPER FUNCTIONS ---
def get_metadata_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        df['PatID'] = df['PatID'].ffill() # Fix merged cells
        meta = {}
        
        def parse_time(val):
            if isinstance(val, (int, float)): return float(val)
            if hasattr(val, 'hour'): return val.hour * 3600 + val.minute * 60 + val.second
            if isinstance(val, str):
                parts = val.split(':')
                if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
            return 0.0

        for _, row in df.iterrows():
            pat_id = str(row['PatID']).strip()
            sz_id = str(row['#Seizure']).strip()
            key = f"{pat_id}_{sz_id}"
            meta[key] = {
                't_eeg_onset': parse_time(row['EEG onset']),
                't_clinical_onset': parse_time(row['Clinical Onset'])
            }
        return meta
    except Exception as e:
        print(f"Excel Error: {e}")
        return {}

# --- 4. MAIN EVALUATION LOOP ---
def run_final_evaluation():
    meta = get_metadata_from_excel(METADATA_EXCEL_PATH)
    if not meta: return None

    with open(MASTER_LABEL_FILE, 'r') as f: 
        all_labels = json.load(f)

    model = VSViG_base().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # List to store detailed dictionary for every seizure
    detailed_records = []

    print(f"{'Seizure ID':<15} | {'LEO':<10} | {'LCO':<10} | {'Status'}")
    print("-" * 65)

    for event_id in sorted(meta.keys()):
        # Extract Patient ID (e.g., "Pat01" from "Pat01_Sz1")
        patient_id = event_id.split('_')[0]

        clips = sorted([k for k in all_labels.keys() if event_id.lower() in k.lower()],
                       key=lambda x: float(x.rsplit('_', 1)[1]))

        if len(clips) < WINDOW_SIZE: continue

        # Calculate Duration of this specific file (for FDR/Hr calculation)
        # Assuming 1 clip = 1 second stride. Adjust if clips represent different time.
        duration_hrs = (len(clips) * STRIDE) / 3600.0

        # Create dummy dict for dataset
        dummy_dict = {clip: 0 for clip in clips}
        temp_list = f"temp_eval_{event_id}.json"
        with open(temp_list, 'w') as f: json.dump(dummy_dict, f)

        # Variables for this seizure
        leo, lco, accum_conf = None, None, 0.0
        status = "FN" # Default to False Negative (Missed)
        t_onset = None

        try:
            dataset = vsvig_dataset(PROCESSED_DATA_DIR, temp_list)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            probs = []
            with torch.no_grad():
                for batch in loader:
                    data, kpts, _, _, _ = batch
                    out = model(data.to(DEVICE), kpts.to(DEVICE))
                    probs.append(float(out.item()))

            # Detection Logic
            max_conf = 0.0
            for i in range(WINDOW_SIZE, len(probs)):
                # Calculate confidence sum in window
                current_conf = sum(probs[i - WINDOW_SIZE : i])
                if current_conf > max_conf: max_conf = current_conf
                
                # Check Threshold
                if current_conf >= DT_SENSITIVITY:
                    t_onset = float(clips[i].rsplit('_', 1)[1])
                    accum_conf = current_conf # Capture confidence at detection
                    break
            
            # If no detection found, store the max confidence seen (for analysis)
            if t_onset is None:
                accum_conf = max_conf

            # Calculate LEO/LCO and Status
            if t_onset:
                leo = t_onset - meta[event_id]['t_eeg_onset']
                lco = t_onset - meta[event_id]['t_clinical_onset']
                
                if lco > PAPER_WINDOW: # > -120s
                    status = "TP" # True Positive
                else:
                    status = "FP" # False Positive (Too Early/False Alarm)
            
            # Print to Console
            lco_str = f"{lco:.1f}" if lco else "None"
            leo_str = f"{leo:.1f}" if leo else "None"
            print(f"{event_id:<15} | {leo_str:>9}s | {lco_str:>9}s | {status}")

            # Append to records
            detailed_records.append({
                "Patient": patient_id,
                "Seizure_ID": event_id,
                "Status": status,
                "Accum_Conf": round(accum_conf, 4),
                "LEO": leo,
                "LCO": lco,
                "Duration_Hrs": duration_hrs
            })

        except Exception as e:
            print(f"Error {event_id}: {e}")
        
        if os.path.exists(temp_list): os.remove(temp_list)

    # Convert to DataFrame
    df_details = pd.DataFrame(detailed_records)
    return df_details

# --- 5. GROUPING & SAVING LOGIC ---
def save_and_summarize(df):
    if df is None or df.empty:
        print("No results to save.")
        return

    # 1. Save Per-Seizure Results
    df.to_csv("per_seizure_results.csv", index=False)
    print("\n✅ Saved detailed results to 'per_seizure_results.csv'")

    # 2. Group by Patient
    patient_stats = []
    
    # Get unique patients
    patients = df['Patient'].unique()

    for pat in patients:
        sub = df[df['Patient'] == pat]
        
        total_seizures = len(sub)
        tp_count = len(sub[sub['Status'] == 'TP'])
        fp_count = len(sub[sub['Status'] == 'FP']) # Early detections
        fn_count = len(sub[sub['Status'] == 'FN']) # Missed
        
        # Duration for FDR calculation
        total_duration = sub['Duration_Hrs'].sum()
        
        # Metrics
        sens = (tp_count / total_seizures) * 100 if total_seizures > 0 else 0
        fdr = fp_count / total_duration if total_duration > 0 else 0
        
        # Mean LEO/LCO (only for True Positives)
        valid_tps = sub[sub['Status'] == 'TP']
        mean_leo = valid_tps['LEO'].mean() if not valid_tps.empty else None
        mean_lco = valid_tps['LCO'].mean() if not valid_tps.empty else None

        patient_stats.append({
            "Patient": pat,
            "Sens": round(sens, 1),
            "FDR_per_Hr": round(fdr, 2),
            "Total_Seizures": total_seizures,
            "TP": tp_count,
            "FN": fn_count,
            "FP_Events": fp_count,
            "Duration_Hrs": round(total_duration, 2),
            "Mean_LEO": round(mean_leo, 2) if mean_leo else None,
            "Mean_LCO": round(mean_lco, 2) if mean_lco else None
        })

    # 3. Save Patient Summary
    df_grouped = pd.DataFrame(patient_stats)
    
    # Reorder columns to match your request
    cols = ["Patient", "Sens", "FDR_per_Hr", "Total_Seizures", 
            "TP", "FN", "FP_Events", "Duration_Hrs", "Mean_LEO", "Mean_LCO"]
    df_grouped = df_grouped[cols]
    
    df_grouped.to_csv("patient_grouping_stats.csv", index=False)
    print("✅ Saved patient stats to 'patient_grouping_stats.csv'")
    
    # Print Preview
    print("\n--- PATIENT SUMMARY PREVIEW ---")
    print(df_grouped.to_string(index=False))

# --- EXECUTE ---
results_df = run_final_evaluation()
save_and_summarize(results_df)