import json
import os
import random
from collections import defaultdict

# --- CONFIG ---
DATA_FOLDER = 'processed_data'
LABEL_FILE = os.path.join(DATA_FOLDER, 'labels.json')
FOLDS_FOLDER = os.path.join(DATA_FOLDER, 'folds')

def main():
    if not os.path.exists(FOLDS_FOLDER):
        os.makedirs(FOLDS_FOLDER)
        
    print(f"📂 Loading labels from {LABEL_FILE}...")
    with open(LABEL_FILE, 'r') as f:
        all_data_dict = json.load(f) # {"pat01_Sz1_1780": 0.0, ...}
        
    # --- GROUPING LOGIC ---
    patient_map = defaultdict(list)
    
    for filename, label in all_data_dict.items():
        # filename: "pat01_Sz1_1780"
        
        # 1. Extract Patient ID (The Grouping Key)
        parts = filename.split('_') 
        patient_id = parts[0] # "pat01"
        
        # 2. Add to that patient's bucket
        patient_map[patient_id].append([filename, label])
        
    # --- VERIFICATION PRINT ---
    print(f"🔎 Found {len(patient_map)} unique patients.")
    for pid, clips in patient_map.items():
        print(f"   👤 {pid}: {len(clips)} clips (Includes all seizures)")

    # --- CREATE FOLDS ---
    all_patients = sorted(patient_map.keys())
    
    for test_patient in all_patients:
        print(f"⚙️ Generating Fold: Test Patient = {test_patient}")
        
        train_clips = []
        val_clips = []
        
        for pid, clips in patient_map.items():
            if pid == test_patient:
                # ALL seizures from this patient go to Validation
                val_clips.extend(clips)
            else:
                # ALL seizures from other patients go to Training
                train_clips.extend(clips)
        
        random.shuffle(train_clips)
        
        # Save
        with open(os.path.join(FOLDS_FOLDER, f'train_{test_patient}.json'), 'w') as f:
            json.dump(train_clips, f)
            
        with open(os.path.join(FOLDS_FOLDER, f'val_{test_patient}.json'), 'w') as f:
            json.dump(val_clips, f)

    print("\n✅ Done! Folds created successfully.")

if __name__ == '__main__':
    main()