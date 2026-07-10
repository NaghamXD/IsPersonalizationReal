import json
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from VSViG import VSViG_base

# --- 1. PATH CONFIGURATION ---
BASE_PATH = "." 
METADATA_EXCEL_PATH = "WU-SAHZU-EMU-Video/dataset/Label.xlsx"
MASTER_LABEL_FILE = os.path.join(BASE_PATH, "processed_data/labels.json")
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, "processed_data")
MODEL_PATH = os.path.join(BASE_PATH, "checkpoints_improved/best_model.pth")
RESULTS_JSON_PATH = "evaluation_report_no_accum_improved.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"✅ Using device: {DEVICE}")

# --- 2. SETTINGS ---
DT_SENSITIVITY = 0.3  # Now treated as a probability threshold (0.0 - 1.0)
STRIDE = 1.0
WINDOW_SIZE = 3 

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
        label = float(self._labels_dict[file_key])
        parts = file_key.split('_')
        unique_id = f"{parts[0]}_{parts[1]}"
        try: frame_num = float(parts[-1]) 
        except: frame_num = 0.0
        
        data_path = os.path.join(self._folder, 'patches', f"{file_key}.pt")
        kpts_path = os.path.join(self._folder, 'kpts', f"{file_key}.pt")
        
        data = torch.load(data_path, map_location='cpu')
        kpts = torch.load(kpts_path, map_location='cpu')
        
        if data.max() > 2.0: data = data.float() / 255.0
        else: data = data.float()
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        data = (data - mean) / std

        kpts = kpts.float()
        if kpts.max() > 2.0:
            kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
            kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        
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
    print(f"🚀 Starting Evaluation (No Accumulation)...")
    
    meta = get_metadata_from_excel(METADATA_EXCEL_PATH)
    with open(MASTER_LABEL_FILE, 'r') as f: all_labels = json.load(f)

    model = VSViG_base().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- PART 1: SEIZURE LATENCY ---
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
                for batch in dl:
                    d = batch[0].to(DEVICE)
                    k = batch[1].to(DEVICE)
                    out = model(d, k)
                    if out.dim() == 0: probs.append(float(out))
                    else: probs.extend(out.cpu().tolist())

            detected = False
            for i in range(WINDOW_SIZE, len(probs)):
                # --- NO ACCUMULATION CHANGE ---
                # Old: sum(probs[i - WINDOW_SIZE : i]) >= DT_SENSITIVITY
                # New: np.mean(probs[i - WINDOW_SIZE : i]) >= DT_SENSITIVITY
                current_score = np.mean(probs[i - WINDOW_SIZE : i])
                
                if current_score >= DT_SENSITIVITY:
                    t_onset = float(clips[i].rsplit('_', 1)[1])
                    lco = t_onset - meta[event_id]['t_clinical_onset']
                    
                    # Ignore if detected WAY too early (before relevant window)
                    if lco < -1200: # e.g. 20 mins before
                        continue

                    leo = t_onset - meta[event_id]['t_eeg_onset']
                    latency_results.append({'id': event_id, 'leo': leo, 'lco': lco})
                    
                    status = "✅ Detected" if lco > -120 else "⚠️ Early"
                    print(f"{event_id:<15} | {leo:>9.1f}s | {lco:>9.1f}s | {status}")
                    detected = True
                    break
            
            if not detected:
                 print(f"{event_id:<15} | {'--':>9} | {'--':>9} | ❌ Missed")

        except Exception as e: print(f"Error on {event_id}: {e}")
        if os.path.exists(temp_file): os.remove(temp_file)

    # --- PART 2: FALSE DETECTION RATE (FDR) ---
    print(f"\n📊 2. Calculating FDR (Buffer: {PRE_ICTAL_BUFFER}s)...")
    
    interictal_clips = []
    skipped_count = 0
    total_clips_checked = 0

    for k, v in all_labels.items():
        total_clips_checked += 1
        if float(v) >= 0.01: continue
            
        prefix = k.rsplit('_', 1)[0]
        if prefix in meta:
            try:
                clip_time = float(k.rsplit('_', 1)[1])
                seizure_start = meta[prefix]['t_eeg_onset']
                time_diff = clip_time - seizure_start
                if -PRE_ICTAL_BUFFER < time_diff < POST_ICTAL_BUFFER:
                    skipped_count += 1
                    continue
            except: continue
        interictal_clips.append(k)

    interictal_clips.sort(key=lambda x: (x.rsplit('_', 1)[0], float(x.rsplit('_', 1)[1])))
    total_hours = (len(interictal_clips) * STRIDE) / 3600.0
    
    print(f"   ℹ️ Analyzing {len(interictal_clips)} clips ({total_hours:.2f} hours).")

    if len(interictal_clips) > 0:
        patient_streams = {}
        for clip in interictal_clips:
            pid = clip.rsplit('_', 1)[0]
            if pid not in patient_streams: patient_streams[pid] = []
            patient_streams[pid].append(clip)

        false_positives = 0
        
        for pid, clips in patient_streams.items():
            if len(clips) < WINDOW_SIZE: continue
            
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
                    if refractory_counter > 0:
                        refractory_counter -= 1
                        continue
                    
                    # --- NO ACCUMULATION CHANGE ---
                    current_score = np.mean(probs[i - WINDOW_SIZE : i])
                    
                    if current_score >= DT_SENSITIVITY:
                        false_positives += 1
                        refractory_counter = int(30.0 / STRIDE) 
                        
            except Exception as e: print(f"FDR Error {pid}: {e}")
            if os.path.exists(temp_file): os.remove(temp_file)

        fdr_per_hour = false_positives / total_hours if total_hours > 0 else 0
        print(f"   Total False Alarms: {false_positives}")
        print(f"   Measured FDR: {fdr_per_hour:.2f} FP/hr")
    else:
        false_positives = 0
        fdr_per_hour = 0

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