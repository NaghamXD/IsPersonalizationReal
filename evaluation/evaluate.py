import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from VSViG import VSViG_base

# --- 1. PATH CONFIGURATION ---
BASE_PATH = "." 
METADATA_EXCEL_PATH = "WU-SAHZU-EMU-Video/dataset/Label.xlsx"
MASTER_LABEL_FILE = os.path.join(BASE_PATH, "processed_data/labels.json")
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, "processed_data")
MODEL_PATH = os.path.join(BASE_PATH, "checkpoints_improved/best_model.pth")
RESULTS_JSON_PATH = "evaluation_report_improved.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✅ Using device: {DEVICE}")

# --- 2. YOUR SETTINGS (Preserved) ---
DT_SENSITIVITY = 0.3
STRIDE = 1.0       # ✅ Kept as you requested
WINDOW_SIZE = 3    # ✅ Kept as you requested

# Configuration for exclusion
PRE_ICTAL_BUFFER = 120   # 2 minutes before seizure
POST_ICTAL_BUFFER = 120  # 2 minutes after seizure

class vsvig_dataset(Dataset):
    def __init__(self, data_folder, label_file):
        with open(label_file, 'r') as f:
            self._labels_dict = json.load(f)
        self._keys = list(self._labels_dict.keys())
        self._folder = data_folder

    def __getitem__(self, idx):
        file_key = self._keys[idx]
        
        # 1. Get Label
        label = float(self._labels_dict[file_key])
        
        # Parse ID
        parts = file_key.split('_')
        unique_id = f"{parts[0]}_{parts[1]}"
        try: 
            # Use float to be safe with both "100" and "100.0"
            frame_num = float(parts[-1]) 
        except: 
            frame_num = 0.0
        
        data_path = os.path.join(self._folder, 'patches', f"{file_key}.pt")
        kpts_path = os.path.join(self._folder, 'kpts', f"{file_key}.pt")
        
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

        # 3. Keypoints Normalization
        kpts = kpts.float()
        if kpts.max() > 2.0:
            kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
            kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        
        # Returns 5 items
        return data, kpts, label, unique_id, frame_num

    def __len__(self):
        return len(self._keys)

def get_metadata_from_excel(excel_path):
    try:
        df = pd.read_excel(excel_path)
        df['PatID'] = df['PatID'].ffill()
        meta = {}
        
        def parse_time(val):
            if isinstance(val, (int, float)): return float(val)
            if hasattr(val, 'hour'): return val.hour * 3600 + val.minute * 60 + val.second
            if isinstance(val, str):
                parts = val.split(':')
                if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
            return 0.0

        for _, row in df.iterrows():
            pat_id = str(row['PatID']).strip().lower()
            sz_id = str(row['#Seizure']).strip()
            key = f"{pat_id}_{sz_id}"
            meta[key] = {
                't_eeg_onset': parse_time(row['EEG onset']),
                't_clinical_onset': parse_time(row['Clinical Onset'])
            }
        return meta
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return {}

def evaluate_model():
    print(f"🚀 Starting Evaluation (Stride={STRIDE}, Window={WINDOW_SIZE})...")
    
    meta = get_metadata_from_excel(METADATA_EXCEL_PATH)
    with open(MASTER_LABEL_FILE, 'r') as f: all_labels = json.load(f)

    model = VSViG_base().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- PART 1: SEIZURE LATENCY (Kept as is) ---
    print("\n📊 1. Calculating Latency...")
    latency_results = []
    seizure_events = sorted(list(set([k.rsplit('_', 1)[0] for k in all_labels.keys()])))
    
    print(f"{'Seizure ID':<15} | {'LEO':<10} | {'LCO':<10} | {'Status'}")
    print("-" * 60)

    for event_id in seizure_events:
        if event_id not in meta: continue

        clips = sorted([k for k in all_labels.keys() if k.startswith(event_id + "_")],
                       key=lambda x: float(x.rsplit('_', 1)[1]))
        
        if len(clips) < WINDOW_SIZE: continue

        temp_dict = {c: 0 for c in clips}
        temp_file = "temp_eval.json"
        with open(temp_file, 'w') as f: json.dump(temp_dict, f)
        
        try:
            ds = vsvig_dataset(PROCESSED_DATA_DIR, temp_file)
            dl = DataLoader(ds, batch_size=32, shuffle=False)
            
            probs = []
            with torch.no_grad():
                # 🔴 ROBUST LOOP: Handles 5-item return
                for batch in dl:
                    d = batch[0].to(DEVICE)
                    k = batch[1].to(DEVICE)
                    out = model(d, k)
                    if out.dim() == 0: probs.append(float(out))
                    else: probs.extend(out.cpu().tolist())

            detected = False
            for i in range(WINDOW_SIZE, len(probs)):
                if sum(probs[i - WINDOW_SIZE : i]) >= DT_SENSITIVITY:
                    t_onset = float(clips[i].rsplit('_', 1)[1])
                    leo = t_onset - meta[event_id]['t_eeg_onset']
                    lco = t_onset - meta[event_id]['t_clinical_onset']
                    
                    # 🔴 CHANGED: We now accept ANY detection, even if very early
                    # This helps debug time offsets
                    latency_results.append({'id': event_id, 'leo': leo, 'lco': lco})
                    
                    if lco > -120:
                        status = "✅ Detected"
                    else:
                        status = "⚠️ Early"
                        
                    print(f"{event_id:<15} | {leo:>9.1f}s | {lco:>9.1f}s | {status}")
                    detected = True
                    break
            
            if not detected:
                 print(f"{event_id:<15} | {'--':>9} | {'--':>9} | ❌ Missed")

        except Exception as e: print(f"Error on {event_id}: {e}")
        if os.path.exists(temp_file): os.remove(temp_file)

    # --- PART 2: FALSE DETECTION RATE (FDR) WITH 2-MIN BUFFER ---
    print(f"\n📊 2. Calculating FDR (Buffer: {PRE_ICTAL_BUFFER}s pre-seizure)...")
    
    interictal_clips = []
    skipped_count = 0
    total_clips_checked = 0

    for k, v in all_labels.items():
        total_clips_checked += 1
        
        # 1. Skip if it is a seizure frame (Label != 0)
        if float(v) >= 0.01:
            continue
            
        # 2. TIME-BASED BUFFER LOGIC
        prefix = k.rsplit('_', 1)[0] # e.g. "Pat1_Sz1"
        
        # If this file belongs to a known seizure event
        if prefix in meta:
            try:
                # Get the current clip time
                clip_time = float(k.rsplit('_', 1)[1])
                
                # Get the exact seizure start time from Excel metadata
                seizure_start = meta[prefix]['t_eeg_onset']
                
                # Calculate distance from seizure
                time_from_onset = clip_time - seizure_start
                
                # EXCLUSION LOGIC:
                # We want to ignore the "Danger Zone" surrounding the seizure.
                # Zone = [Start - 2min]  to  [Start + 2min]
                # If time_from_onset is -60 (1 min before), we skip.
                # If time_from_onset is -300 (5 mins before), we keep it.
                if -PRE_ICTAL_BUFFER < time_from_onset < POST_ICTAL_BUFFER:
                    skipped_count += 1
                    continue
                    
            except Exception as e:
                # If metadata is missing/malformed, safe default is to skip to avoid errors
                continue

        # If we survive the checks, add to analysis list
        interictal_clips.append(k)

    # Sort clips to ensure we process time sequentially
    interictal_clips.sort(key=lambda x: (x.rsplit('_', 1)[0], float(x.rsplit('_', 1)[1])))
    
    total_hours = (len(interictal_clips) * STRIDE) / 3600.0
    
    print(f"   ℹ️ Total clips checked: {total_clips_checked}")
    print(f"   ℹ️ Skipped {skipped_count} clips inside the 2-min buffer zone.")
    print(f"   ✅ Analyzing {len(interictal_clips)} clips ({total_hours:.2f} hours).")

    if len(interictal_clips) == 0:
        print("❌ CRITICAL: No background data found! FDR will be 0.")
        # Create empty report to avoid crash
        false_positives = 0
        fdr_per_hour = 0
    else:
        # Group by patient for efficient processing
        patient_streams = {}
        for clip in interictal_clips:
            pid = clip.rsplit('_', 1)[0]
            if pid not in patient_streams: patient_streams[pid] = []
            patient_streams[pid].append(clip)

        false_positives = 0
        
        for pid, clips in patient_streams.items():
            if len(clips) < WINDOW_SIZE: continue
            
            # Create temp JSON for just this patient's background data
            temp_dict = {c: 0 for c in clips}
            temp_file = "temp_fdr.json"
            with open(temp_file, 'w') as f: json.dump(temp_dict, f)
            
            try:
                ds = vsvig_dataset(PROCESSED_DATA_DIR, temp_file)
                dl = DataLoader(ds, batch_size=32, shuffle=False)
                
                probs = []
                with torch.no_grad():
                    for batch in dl:
                        d = batch[0].to(DEVICE)
                        k = batch[1].to(DEVICE)
                        out = model(d, k)
                        if out.dim() == 0: probs.append(float(out))
                        else: probs.extend(out.cpu().tolist())

                refractory_counter = 0
                for i in range(WINDOW_SIZE, len(probs)):
                    # Refractory logic: Don't count multiple alarms for the same event
                    if refractory_counter > 0:
                        refractory_counter -= 1
                        continue
                    
                    # If probability > threshold, it's a False Alarm
                    if sum(probs[i - WINDOW_SIZE : i]) >= DT_SENSITIVITY:
                        false_positives += 1
                        refractory_counter = int(30.0 / STRIDE) # 30s silence after alarm
                        
            except Exception as e: print(f"FDR Error {pid}: {e}")
            if os.path.exists(temp_file): os.remove(temp_file)

        fdr_per_hour = false_positives / total_hours if total_hours > 0 else 0
        print(f"   Total False Alarms: {false_positives}")
        print(f"   Measured FDR: {fdr_per_hour:.2f} FP/hr")

    # --- 3. SAVE REPORT ---
    report = {
        "config": {"stride": STRIDE, "window": WINDOW_SIZE, "dt": DT_SENSITIVITY},
        "latency_metrics": {
            "avg_leo": float(np.mean([x['leo'] for x in latency_results])) if latency_results else 0,
            "avg_lco": float(np.mean([x['lco'] for x in latency_results])) if latency_results else 0,
            "sensitivity": len(latency_results) / len(seizure_events) if seizure_events else 0
        },
        "fdr_metrics": {
            "total_false_positives": false_positives,
            "total_hours_analyzed": total_hours,
            "fdr_per_hour": fdr_per_hour
        },
        "detailed_results": latency_results
    }
    
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ Report saved to {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    evaluate_model()