from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
import argparse

# --- CONFIG ---
DATA_FOLDER = 'processed_data'
FOLDS_FOLDER = os.path.join(DATA_FOLDER, 'folds')
MODEL_SAVE_ROOT = 'checkpoints_lopo'

# Default ID if running without arguments
DEFAULT_TEST_PATIENT = 'Pat03'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_patient', type=str, default=DEFAULT_TEST_PATIENT, 
                        help='The Patient ID to use as validation set (e.g., Pat01)')
    return parser.parse_args()

# --- DATASET CLASS (UPDATED) ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        self._folder = data_folder
        self._transform = transform
        
        # Define subfolders for patches and keypoints
        self.patches_dir = os.path.join(data_folder, "patches")
        self.kpts_dir = os.path.join(data_folder, "kpts")
        
        # Check if folders exist
        if not os.path.exists(self.patches_dir) or not os.path.exists(self.kpts_dir):
            raise FileNotFoundError(f"Ensure 'patches' and 'kpts' folders exist inside {data_folder}")

        # Load Labels
        with open(label_file, 'rb') as f:
            self._labels = json.load(f)

    def __getitem__(self, idx):
        # Assuming label structure: [filename_string, label_value]
        # Example: ["pat01_Sz1_1780", 1]
        filename_base = str(self._labels[idx][0])
        target = float(self._labels[idx][1])
        
        # Construct paths
        # NOTE: Assuming files end in .pt. If they represent .npy files, change to .npy and use np.load
        patch_path = os.path.join(self.patches_dir, f"{filename_base}.pt")
        kpts_path = os.path.join(self.kpts_dir, f"{filename_base}.pt")
        
        # Load Data
        try:
            data = torch.load(patch_path, map_location='cpu') # Shape: (30, 15, 3, 32, 32)
            kpts = torch.load(kpts_path, map_location='cpu')  # Shape: (30, 15, 2)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find files for ID: {filename_base} at {patch_path}")
        
        # --- PREVIOUS FIX 1: NORMALIZE KEYPOINTS ---
        # Ensure kpts are float
        kpts = kpts.float()
          
        kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
        kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        
        '''
        # --- PREVIOUS FIX 2: ADD CONFIDENCE CHANNEL (2 -> 3 CHANNELS) ---
        # Current shape: (30, 15, 2) -> We need: (30, 15, 3)
        confidence = torch.ones((30, 15, 1), dtype=kpts.dtype)
        kpts = torch.cat((kpts, confidence), dim=2)
        
        if self._transform: 
            # Flatten dimensions for transform if needed, then reshape back
            if len(data.shape) == 5:
                B_frames, P, C, H, W = data.shape 
                data = data.view(B_frames*P*C, H, W)
                data = self._transform(data)
                data = data.view(B_frames, P, C, H, W)
         '''    
        sample = {
            'data': data,
            'kpts': kpts 
        }
        return sample, target
    
    def __len__(self):
        return len(self._labels)

def train():
    args = parse_args()
    test_patient = args.test_patient
    
    # Dynamic Paths based on Patient ID
    train_label_file = os.path.join(FOLDS_FOLDER, f'train_{test_patient}.json')
    val_label_file = os.path.join(FOLDS_FOLDER, f'val_{test_patient}.json')
    
    # Separate Checkpoint Folder for this Fold
    fold_checkpoint_dir = os.path.join(MODEL_SAVE_ROOT, test_patient)
    if not os.path.exists(fold_checkpoint_dir):
        os.makedirs(fold_checkpoint_dir)

    path_to_last_ckpt = os.path.join(fold_checkpoint_dir, 'last_checkpoint.pth')
    path_to_best_model = os.path.join(fold_checkpoint_dir, 'best_model.pth')
    path_to_log = os.path.join(fold_checkpoint_dir, 'training_log.json')    
    
    # 1. Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Fold: Leave-One-Out ({test_patient})")
    print(f"📂 Train File: {train_label_file}")
    print(f"📂 Val File:   {val_label_file}")
    
    # 2. Dataset
    train_dataset = vsvig_dataset(DATA_FOLDER, train_label_file)
    val_dataset = vsvig_dataset(DATA_FOLDER, val_label_file)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Model
    model = VSViG_base().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    MSE = nn.MSELoss()

    # 4. Resume Logic
    start_epoch = 0
    min_valid_loss = np.inf
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    if os.path.exists(path_to_last_ckpt):
        print(f"🔄 Resuming {test_patient} from Epoch...")
        checkpoint = torch.load(path_to_last_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_valid_loss = checkpoint['min_valid_loss']
        history = checkpoint.get('history', history)
    
    # 5. Training Loop
    epochs = 200
    # Add before training loop
    patience = 3
    trigger_times = 0

    for e in range(start_epoch, epochs):
        train_loss = 0.0
        model.train()
        
        print(f'\n=== Epoch {e+1}/{epochs} ===')
        for batch_idx, (sample, labels) in enumerate(train_loader):
            data = sample['data'].to(device)
            kpts = sample['kpts'].to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(data, kpts)
            if outputs.dim() > 1: outputs = outputs.squeeze(1)
            
            loss = MSE(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss (Avg): {avg_train_loss:.4f}")
        history['train_loss'].append(avg_train_loss)

        # Validation
        if (e+1) % 5 == 0:
            valid_loss = 0.0
            rmse_accum = 0.0
            count = 0
            model.eval()
            with torch.no_grad():
                for sample, labels in val_loader:
                    data = sample['data'].to(device)
                    kpts = sample['kpts'].to(device)
                    labels = labels.float().to(device)
                    
                    outputs = model(data, kpts)
                    if outputs.dim() > 1: outputs = outputs.squeeze(1)
                    
                    loss_val = MSE(outputs, labels)
                    valid_loss += loss_val.item()
                    rmse_accum += torch.sqrt(loss_val).item() * 100
                    count += 1
            
            avg_val_loss = valid_loss / count
            avg_rmse = rmse_accum / count
            print(f' >> Val Loss: {avg_val_loss:.4f} | Val RMSE: {avg_rmse:.2f}')
            
            if avg_val_loss < min_valid_loss:
                print(f' >> ✅ New Best for {test_patient}!')
                min_valid_loss = avg_val_loss
                torch.save(model.state_dict(), path_to_best_model)
                trigger_times = 0  # Reset patience counter
            else:
                trigger_times += 1
                print(f' >> ❌ No Improvement. Trigger Times: {trigger_times}/{patience}')
                if trigger_times >= patience:
                    print(f' >> ⏸️ Early stopping triggered for {test_patient} at epoch {e+1}.')
                    break  # Exit training loop
            history['val_loss'].append(avg_val_loss)
            history['val_rmse'].append(avg_rmse)

        scheduler.step()

        # Save Checkpoint per fold
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, path_to_last_ckpt)
        
        with open(path_to_log, 'w') as f:
            json.dump(history, f, indent=4)

if __name__ == '__main__':
    train()