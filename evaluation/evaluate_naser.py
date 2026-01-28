import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from VSViG import VSViG_base


# --- 1. PATH CONFIGURATION ---
# Based on the paths you provided
BASE_PATH = "/Users/naghamdaood/Projects/VSViG" 

# Metadata
METADATA_EXCEL_PATH = "/Users/naghamdaood/Projects/VSViG/WU-SAHZU-EMU-Video/dataset/Label.xlsx"
MASTER_LABEL_FILE = os.path.join(BASE_PATH, "processed_data/labels.json")

# Data Directories (Used by your Dataset Class internally, but good to define)
KPTS_DIR = os.path.join(BASE_PATH, "processed_data/kpts")
PATCHES_DIR = os.path.join(BASE_PATH, "processed_data/patches")
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, "processed_data")

# Model (Update this to your actual .pth file location)
MODEL_PATH = os.path.join(BASE_PATH, "checkpoints/best_model.pth") # <--- UPDATE THIS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. SETTINGS ---
DT_SENSITIVITY = 0.3
STRIDE = 1.0
WINDOW_SIZE = 3

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
        
        # --- KEYPOINT NORMALIZATION FIX ---
        kpts = kpts.float()
        kpts[:, :, 0] /= 1920.0
        kpts[:, :, 1] /= 1080.0
        
        return data, kpts, label, unique_id, frame_num

    def __len__(self):
        return len(self._keys)
    
# --- 3. HELPER: CONVERT EXCEL TO DICT ---
def get_metadata_from_excel(excel_path):
    """
    Reads the Excel file, handles merged cells, converts timestamps 
    to seconds, and builds the dictionary.
    """
    try:
        # 1. Read Excel
        df = pd.read_excel(excel_path)
        
        # 2. Fix Merged/Empty Cells (Forward Fill)
        # This ensures 'Sz2' knows it belongs to 'Pat01' even if the cell is empty
        df['PatID'] = df['PatID'].ffill()

        meta = {}
        
        # 3. Helper to convert HH:MM:SS to Total Seconds
        def parse_time(val):
            # If it's already a number, return it
            if isinstance(val, (int, float)):
                return float(val)
            
            # If it's a datetime object (Pandas sometimes does this auto-magically)
            if hasattr(val, 'hour'):
                return val.hour * 3600 + val.minute * 60 + val.second
            
            # If it's a string like "0:59:40"
            if isinstance(val, str):
                parts = val.split(':')
                if len(parts) == 3: # H:M:S
                    return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                elif len(parts) == 2: # M:S
                    return int(parts[0])*60 + int(parts[1])
            
            return 0.0 # Fallback

        # 4. Iterate and Build Dictionary
        for _, row in df.iterrows():
            # Construct Key: "Pat01_Sz1" to match your filenames
            # We strip whitespace just in case
            pat_id = str(row['PatID']).strip()
            sz_id = str(row['#Seizure']).strip()
            
            # Combined key (e.g., "Pat01_Sz1")
            key = f"{pat_id}_{sz_id}"
            
            meta[key] = {
                't_eeg_onset': parse_time(row['EEG onset']),
                't_clinical_onset': parse_time(row['Clinical Onset'])
            }
            
        print(f"Successfully loaded metadata for {len(meta)} seizures.")
        # Debug: Print first item to verify
        first_key = list(meta.keys())[0]
        print(f"Example - {first_key}: {meta[first_key]}")
        
        return meta

    except Exception as e:
        print(f"Error reading Excel: {e}")
        return {}
    
# --- 4. MAIN EVALUATION LOOP ---
# Helper path variable (Add this above the function if needed)
# This points to the folder containing both 'patches' and 'kpts'
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, "processed_data")

def run_unfiltered_final_loop():
    meta = get_metadata_from_excel(METADATA_EXCEL_PATH)
    if not meta: return []

    with open(MASTER_LABEL_FILE, 'r') as f: 
        all_labels = json.load(f)

    model = VSViG_base().to(DEVICE) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results_table = []
    
    print(f"{'Seizure ID':<15} | {'LEO':<10} | {'LCO':<10} | {'Status'}")
    print("-" * 60)

    for event_id in sorted(meta.keys()):
        # Find clips
        clips = sorted([k for k in all_labels.keys() if event_id.lower() in k.lower()],
                       key=lambda x: float(x.rsplit('_', 1)[1]))

        if len(clips) < WINDOW_SIZE: continue

        # --- FIX 1: Create a Dummy Dictionary, not a List ---
        # The dataset class expects a dict to look up labels. 
        # We give it dummy '0' labels since we are just predicting.
        dummy_dict = {clip: 0 for clip in clips}
        
        temp_list = f"temp_eval_{event_id}.json"
        with open(temp_list, 'w') as f: json.dump(dummy_dict, f)
        
        try:
            # --- FIX 2: Pass PROCESSED_DATA_DIR (Parent Folder) ---
            # Your Dataset class adds '/patches' and '/kpts' automatically.
            dataset = vsvig_dataset(PROCESSED_DATA_DIR, temp_list)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            probs = []
            with torch.no_grad():
                for batch in loader:
                    # --- FIX 3: Correct Unpacking ---
                    # Your dataset returns 5 items, not 2.
                    data_tensor, kpts_tensor, label, unique_id, frame_num = batch
                    
                    # Pass correct tensors to model
                    out = model(data_tensor.to(DEVICE), kpts_tensor.to(DEVICE))
                    probs.append(float(out.item()))
            
            # Logic for Detection
            t_onset = None
            for i in range(WINDOW_SIZE, len(probs)):
                if sum(probs[i - WINDOW_SIZE : i]) >= DT_SENSITIVITY:
                    t_onset = float(clips[i].rsplit('_', 1)[1])
                    break

            if t_onset:
                leo = t_onset - meta[event_id]['t_eeg_onset']
                lco = t_onset - meta[event_id]['t_clinical_onset']
                results_table.append((event_id, leo, lco)) 
                status = "PRE ✅" if lco < 0 else "POST ⚠️"
                print(f"{event_id:<15} | {leo:>9.1f}s | {lco:>9.1f}s | {status}")
            else:
                print(f"{event_id:<15} | {'No Det':>10} | {'No Det':>10} | (Below Threshold)")

        except Exception as e:
            # Print full error details to help debug if it happens again
            import traceback
            print(f"Error processing {event_id}: {e}")
            traceback.print_exc()
        
        # Cleanup
        if os.path.exists(temp_list):
            os.remove(temp_list)

    return results_table

# --- 5. EXECUTION ---
# Run the evaluation
all_evaluation_results = run_unfiltered_final_loop()

# --- 6. SUMMARY STATS ---
def run_paper_correct_summary(results_list):
    # results_list structure: (ID, LEO, LCO)
    # Filter valid detections (Association window > -120s)
    valid_detections = [r for r in results_list if r[2] > -120] # Using LCO for filtering
    false_alarms = [r for r in results_list if r[2] <= -120]

    clean_leo = [r[1] for r in valid_detections]
    clean_lco = [r[2] for r in valid_detections]

    print("\n" + "="*40)
    print("🏆 FINAL VSViG PAPER COMPARISON")
    print("="*40)
    print(f"{'Metric':<15} | {'Your Model':<10} | {'Paper Goal'}")
    print("-" * 40)
    
    if clean_leo:
        print(f"{'Avg LEO':<15} | {np.mean(clean_leo):>8.2f}s | ~5.1s")
        print(f"{'Avg LCO':<15} | {np.mean(clean_lco):>8.2f}s | ~-13.1s")
        sensitivity = len(valid_detections) / 33 * 100 # Assuming 33 total patients
        print(f"{'Sensitivity':<15} | {sensitivity:>8.1f}% | >85%")
    
    print(f"{'False Alarms':<15} | {len(false_alarms):>10} | (Report as FDR)")
    print("-" * 40)

run_paper_correct_summary(all_evaluation_results)

# --- 7. PLOTTING (Automatic) ---

if all_evaluation_results:
    # 1. Get Data from previous step
    # Tuple is (ID, LEO, LCO)
    valid_data = [r for r in all_evaluation_results if r[2] > -120]
    valid_data.sort(key=lambda x: x[1]) # Sort by LEO

    ids = [item[0] for item in valid_data]
    leos = [item[1] for item in valid_data]
    lcos = [item[2] for item in valid_data]

    # 2. Create Visualization
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Primary Axis: LEO
    bars = ax1.bar(ids, leos, color='skyblue', alpha=0.8, label='Your LEO (Latency)')
    ax1.set_ylabel('Latency from EEG Onset ($LEO$ in Seconds)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Seizure ID', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=90)

    # Benchmark LEO
    ax1.axhline(y=5.1, color='blue', linestyle='--', linewidth=2, label='VSViG Paper Goal (5.1s)')

    # Secondary Axis: LCO
    ax2 = ax1.twinx()
    ax2.plot(ids, lcos, color='red', marker='o', linestyle='-', linewidth=1.5, markersize=6, label='Your LCO (Warning)')
    ax2.set_ylabel('Warning Time from Clinical Onset ($LCO$ in Seconds)', color='red', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    # Benchmark LCO
    ax2.axhline(y=-13.1, color='red', linestyle='--', linewidth=2, label='VSViG Paper Goal (-13.1s)')

    # Styling
    plt.title('Clinical Performance: VSViG Model vs. Paper Benchmarks', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=10)

    # Stats Box
    stats_text = (
        f"SUMMARY METRICS:\n"
        f"Avg LEO: {np.mean(leos):.2f}s (Goal: 5.1s)\n"
        f"Avg LCO: {np.mean(lcos):.2f}s (Goal: -13.1s)\n"
        f"Sensitivity: {len(valid_data)/33*100:.1f}%\n"
        f"Sync Outliers: {len(all_evaluation_results)-len(valid_data)}"
    )
    plt.text(1.02, 0.5, stats_text, transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('clinical_results_final.png')
    print("Plot saved as clinical_results_final.png")
    plt.show()
else:
    print("No results to plot.")