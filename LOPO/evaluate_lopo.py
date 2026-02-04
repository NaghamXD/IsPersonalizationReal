import torch
import numpy as np
import pandas as pd
import json
import os
import re
from torch.utils.data import Dataset, DataLoader
from VSViG import VSViG_base

# --- CONFIGURATION ---
DATA_FOLDER = 'processed_data'
FOLDS_FOLDER = os.path.join(DATA_FOLDER, 'folds')
CHECKPOINT_ROOT = 'checkpoints'
EXCEL_FILE = 'WU-SAHZU-EMU-Video/dataset/Label.xlsx' 

# Paper Parameters
FPS = 25.0
TAU = 3.0   # Accumulation Window
DT = 0.3    # Decision Threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- HELPER FUNCTIONS (Same as before) ---
def time_str_to_seconds(time_val):
    if pd.isna(time_val): return -1
    if hasattr(time_val, 'hour'): return time_val.hour * 3600 + time_val.minute * 60 + time_val.second
    t_str = str(time_val).strip()
    try:
        parts = t_str.split(':')
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
    except: return -1
    return -1

def load_ground_truth(file_path):
    if file_path.endswith('.xlsx'): df = pd.read_excel(file_path)
    else: df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    gt_map = {}
    for _, row in df.iterrows():
        pid = str(row['PatID']).strip()
        sid = str(row['#Seizure']).strip()
        gt_map[f"{pid}_{sid}".lower()] = {
            'eeg': time_str_to_seconds(row['EEG onset']),
            'clinical': time_str_to_seconds(row['Clinical Onset'])
        }
    return gt_map

class vsvig_dataset(Dataset):
    def __init__(self, data_folder, label_list):
        self._labels = label_list # List of [filename, label]
        self._folder = data_folder

    def __getitem__(self, idx):
        item = self._labels[idx]
        file_key = item[0]
        label = float(item[1])
        
        parts = file_key.split('_')
        unique_id = f"{parts[0]}_{parts[1]}"
        try: frame_num = int(parts[-1])
        except: frame_num = 0
        
        # Path Logic
        data_path = os.path.join(self._folder, 'patches', f"{file_key}.pt")
        kpts_path = os.path.join(self._folder, 'kpts', f"{file_key}.pt")
        if not os.path.exists(data_path):
            data_path = os.path.join(self._folder, f"{file_key}.pt")
            kpts_path = os.path.join(self._folder, f"{file_key}_kpts.pt")
            
        data = torch.load(data_path, map_location='cpu')
        kpts = torch.load(kpts_path, map_location='cpu').float()
        kpts[:, :, 0] /= 1920.0
        kpts[:, :, 1] /= 1080.0
        return data, kpts, label, unique_id, frame_num

    def __len__(self): return len(self._labels)

def calculate_accumulative_decision(frames, scores, fps, tau, threshold):
    if not frames: return False, None, 0.0
    sorted_pairs = sorted(zip(frames, scores), key=lambda x: x[0])
    sorted_frames = [x[0] for x in sorted_pairs]
    sorted_scores = [x[1] for x in sorted_pairs]
    
    max_accum_score = 0.0
    detection_frame = None
    
    for i in range(len(sorted_frames)):
        current_frame = sorted_frames[i]
        current_time = current_frame / fps
        start_time_window = current_time - tau
        
        window_scores = []
        for j in range(i, -1, -1):
            if (sorted_frames[j] / fps) < start_time_window: break
            window_scores.append(sorted_scores[j])
            
        ap = sum(window_scores) / len(window_scores) if window_scores else 0.0
        max_accum_score = max(max_accum_score, ap)
        
        if ap > threshold and detection_frame is None:
            detection_frame = current_frame
            
    return (detection_frame is not None), detection_frame, max_accum_score

# --- MAIN LOPO EVALUATION ---
def main():
    gt_map = load_ground_truth(EXCEL_FILE)
    
    # Identify all Test Patients by looking at folds folder
    val_files = [f for f in os.listdir(FOLDS_FOLDER) if f.startswith('val_') and f.endswith('.json')]
    # Sort nicely (val_Pat01, val_Pat02...)
    val_files.sort()
    
    print(f"🔎 Found {len(val_files)} folds to evaluate.")
    
    # Global Stats accumulators
    global_tp = 0
    global_fn = 0
    global_fp = 0
    global_sz_count = 0
    all_seizure_details = []
    patient_summaries = []

    # === LOOP THROUGH EVERY PATIENT ===
    for v_file in val_files:
        # Extract ID: "val_pat01.json" -> "pat01"
        patient_id = v_file.replace('val_', '').replace('.json', '')
        
        print(f"\n==========================================")
        print(f"🩺 Evaluating Fold: {patient_id}")
        print(f"==========================================")
        
        # 1. Define Model Path for THIS patient
        model_path = os.path.join(CHECKPOINT_ROOT, patient_id, 'best.pth') # Or 'best_model.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}. Skipping this patient.")
            continue
            
        # 2. Load Model
        model = VSViG_base().to(DEVICE)
        ckpt = torch.load(model_path, map_location=DEVICE)
        # Handle state dict vs full checkpoint
        if 'model_state_dict' in ckpt: model.load_state_dict(ckpt['model_state_dict'])
        else: model.load_state_dict(ckpt)
        model.eval()
        
        # 3. Load Validation Data for THIS patient
        val_path = os.path.join(FOLDS_FOLDER, v_file)
        with open(val_path, 'r') as f:
            val_labels = json.load(f)
            
        dataset = vsvig_dataset(DATA_FOLDER, val_labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 4. Run Inference
        events = {}
        with torch.no_grad():
            for data, kpts, labels, unique_ids, frame_nums in loader:
                data = data.to(DEVICE)
                kpts = kpts.to(DEVICE)
                outputs = model(data, kpts)
                if outputs.dim() > 1: outputs = outputs.squeeze(1)
                
                for i in range(len(outputs)):
                    uid = unique_ids[i].lower()
                    score = outputs[i].item()
                    true_label = float(labels[i])
                    frame = frame_nums[i].item()
                    
                    if uid not in events: events[uid] = {'scores': [], 'frames': [], 'is_seizure': False}
                    events[uid]['scores'].append(score)
                    events[uid]['frames'].append(frame)
                    if true_label > 0: events[uid]['is_seizure'] = True

        # 5. Calculate Metrics for THIS Patient
        p_tp, p_fn, p_fp, p_sz = 0, 0, 0, 0
        p_leos, p_lcos = [], []
        
        for uid, data in events.items():
            gt_info = gt_map.get(uid, {'eeg': -1, 'clinical': -1})
            is_detected, detect_frame, max_conf = calculate_accumulative_decision(
                data['frames'], data['scores'], FPS, TAU, DT
            )
            
            if data['is_seizure']:
                p_sz += 1
                if is_detected:
                    p_tp += 1
                    status = "TP"
                    d_time = detect_frame / FPS
                    leo = d_time - gt_info['eeg'] if gt_info['eeg'] != -1 else None
                    lco = d_time - gt_info['clinical'] if gt_info['clinical'] != -1 else None
                    if leo: p_leos.append(leo)
                    if lco: p_lcos.append(lco)
                else:
                    p_fn += 1
                    status = "FN"
                    leo, lco = None, None
                
                all_seizure_details.append({
                    "Patient": patient_id, "Seizure_ID": uid, "Status": status,
                    "Conf": max_conf, "LEO": leo, "LCO": lco
                })
            else:
                if is_detected: p_fp += 1
        
        # Store Patient Summary
        sensitivity = p_tp / p_sz if p_sz > 0 else 0
        print(f"   -> Found {p_sz} Seizures. Detected: {p_tp}. Sensitivity: {sensitivity:.2%}")
        
        patient_summaries.append({
            "Patient": patient_id, "TP": p_tp, "FN": p_fn, "FP": p_fp, "Total_Sz": p_sz,
            "Sens": sensitivity, "Mean_LEO": np.mean(p_leos) if p_leos else None
        })
        
        # Add to Globals
        global_tp += p_tp
        global_fn += p_fn
        global_fp += p_fp
        global_sz_count += p_sz

    # === FINAL AGGREGATE REPORT ===
    print("\n" + "="*50)
    print("       FINAL DATASET RESULTS (LOPO)       ")
    print("="*50)
    
    overall_sens = global_tp / global_sz_count if global_sz_count > 0 else 0
    
    print(f"TOTAL SEIZURES: {global_sz_count}")
    print(f"TOTAL DETECTED: {global_tp}")
    print(f"TOTAL MISSED:   {global_fn}")
    print(f"TOTAL FALSE ALARMS: {global_fp}")
    print(f"📈 OVERALL SENSITIVITY: {overall_sens:.2%}")
    print("="*50)
    
    # Save CSVs
    pd.DataFrame(all_seizure_details).to_csv("FINAL_seizure_report.csv", index=False)
    pd.DataFrame(patient_summaries).to_csv("FINAL_patient_summary.csv", index=False)
    print("✅ Saved 'FINAL_seizure_report.csv' and 'FINAL_patient_summary.csv'")

if __name__ == '__main__':
    main()