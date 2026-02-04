from VSViG import *
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
import torch.nn.functional as F  # Added for manual RMSE calculation


# --- CONFIGURATION ---
PROCESSED_FOLDER = "processed_data" 
PATH_TO_DATA_FOLDER = PROCESSED_FOLDER 

# --- PATH CONFIGS ---
CHECKPOINT_DIR = "checkpoints_improved"
if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

# --- DATASET CLASS (UPDATED) ---
class vsvig_dataset(Dataset):
    def __init__(self, data_folder=None, label_file=None, transform=None):
        super().__init__()
        
        self.patches_dir = os.path.join(data_folder, "patches")
        self.kpts_dir = os.path.join(data_folder, "kpts")
        
        if not os.path.exists(self.patches_dir) or not os.path.exists(self.kpts_dir):
            raise FileNotFoundError(f"Ensure 'patches' and 'kpts' folders exist inside {data_folder}")

        # 🔴 SMART LOADER: Handles both Dict and List 🔴
        with open(label_file, 'r') as f:
            raw_data = json.load(f)
            
        if isinstance(raw_data, list):
            # Case A: It's already a List (Your train_labels.json)
            print(f"✅ Detected LIST format in {os.path.basename(label_file)}")
            self._labels = raw_data
        elif isinstance(raw_data, dict):
            # Case B: It's a Dictionary (The master labels.json)
            print(f"✅ Detected DICT format in {os.path.basename(label_file)} (Converting...)")
            self._labels = [[k, v] for k, v in raw_data.items()]
        else:
            raise ValueError("Label file must be a JSON List or Dictionary.")
            
        print(f"   Loaded {len(self._labels)} samples.")

    def __getitem__(self, idx):
        # Access list item
        filename_base = str(self._labels[idx][0])
        target = float(self._labels[idx][1])
        
        patch_path = os.path.join(self.patches_dir, f"{filename_base}.pt")
        kpts_path = os.path.join(self.kpts_dir, f"{filename_base}.pt")
        
        try:
            data = torch.load(patch_path, map_location='cpu') # Shape: (30, 15, 3, 32, 32)
            kpts = torch.load(kpts_path, map_location='cpu')  # Shape: (30, 15, 2)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find files for ID: {filename_base}")
        
        # -----------------------------------------------------------
        # [FIX 2] NORMALIZATION & STANDARDIZATION
        # -----------------------------------------------------------
        
        # 1. Ensure float and [0, 1] range
        # Your preprocessing saves as [0, 1], so we typically don't need to divide.
        # But just in case any 0-255 crept in:
        if data.max() > 2.0: 
            data = data.float() / 255.0
        else:
            data = data.float()
            
        # 2. Apply Standardization (Mean/Std)
        # This centers the data around 0, which is critical for the ReLU layers to work well.
        # Create Mean/Std tensors that match the Channel dimension (Dim 2)
        # Data shape: (30, 15, 3, 32, 32)
        # We reshape mean/std to (1, 1, 3, 1, 1) to broadcast correctly
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        
        # PyTorch handles the broadcasting automatically
        data = (data - mean) / std

        # 3. Keypoints Normalization [0, 1]
        kpts = kpts.float()
        if kpts.max() > 2.0:
            kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
            kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
            
        sample = {'data': data, 'kpts': kpts}
        return sample, target
    
    def __len__(self):
        return len(self._labels)

# --- CUSTOM WEIGHTED LOSS (Optional Alternative) ---
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        # If target is seizure (>0.5), multiply error by pos_weight
        weights = torch.ones_like(loss)
        weights[target > 0.5] = self.pos_weight 
        return (loss * weights).mean()
    
def train():
    train_label_path = os.path.join(PROCESSED_FOLDER, 'train_labels.json')
    val_label_path = os.path.join(PROCESSED_FOLDER, 'val_labels.json')
    
    print("🚀 Starting Advanced Training Pipeline...")
    # 1. Setup Data
    dataset_train = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=train_label_path)
    dataset_val = vsvig_dataset(data_folder=PATH_TO_DATA_FOLDER, label_file=val_label_path)

    # Using smaller batch size is often better for generalization in GNNs
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=8)

    # 2. Setup Model
    model = VSViG_base() # Using your clean architecture
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Device: {device}")
    model = model.to(device)
    
    # 3. Loss Function (Huber is safer than MSE)
    # Delta=1.0 means it acts like MSE for errors < 1.0 and MAE for errors > 1.0
    criterion = nn.HuberLoss(delta=1.0)
    
    # OPTIONAL: Use Weighted MSE if your dataset has very few seizures
    # criterion = WeightedMSELoss(pos_weight=10.0) 
    
    # 4. Optimizer (AdamW is standard for ViTs/GNNs)
    # Weight decay helps prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # 5. Scheduler (Cosine Annealing with Warm Restarts)
    # This restarts the LR every 10 epochs, helping find better global minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    min_valid_loss = np.inf
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    # Check for Resume
    start_epoch = 0
    if os.path.exists(PATH_TO_LAST_CKPT):
        print(f"Found checkpoint: {PATH_TO_LAST_CKPT}. Resuming...")
        checkpoint = torch.load(PATH_TO_LAST_CKPT, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_valid_loss = checkpoint['min_valid_loss']
        
        if 'history' in checkpoint:
            history = checkpoint['history']

    elif os.path.exists(PATH_TO_BEST_MODEL):
        print(f"Found best_model.pth but no checkpoint. Loading weights only.")
        try:
            model.load_state_dict(torch.load(PATH_TO_BEST_MODEL, map_location=device))
        except:
            print("Could not load best_model.pth weights. Starting fresh.")

    epochs = 200
    
    # 3. Training Loop
    for e in range(start_epoch, epochs):
        train_loss = 0.0
        model.train()

        print(f'\n=== Epoch: {e+1}/{epochs} [LR: {optimizer.param_groups[0]["lr"]:.6f}] ===')

        for batch_idx, (sample, labels) in enumerate(train_loader):
            data = sample['data'].to(device)
            kpts = sample['kpts'].to(device)
            labels = labels.float().to(device)

            outputs = model(data, kpts)
            
            # Ensure shapes match (B) vs (B,1)
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 🔴 GRADIENT CLIPPING (Crucial stability fix)
            # Prevents "exploding gradients" which ruin training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"\rBatch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f'\n📉 Train Loss: {avg_train_loss:.4f}')

        # 4. Validation & Saving
        if (e+1) % 5 == 0:
            valid_loss = 0.0
            rmse_accum = 0.0
            model.eval()
            
            with torch.no_grad():
                for sample, labels in val_loader:
                    data = sample['data'].to(device)
                    kpts = sample['kpts'].to(device)
                    labels = labels.float().to(device)
                    
                    outputs = model(data, kpts)
                    if outputs.dim() > 1 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                    
                    # Huber Loss for optimization tracking
                    valid_loss += criterion(outputs, labels).item()
                    
                    # RMSE for human readability
                    # We use F.mse_loss to calculate MSE, then sqrt
                    rmse_accum += torch.sqrt(F.mse_loss(outputs, labels)).item() * 100
            
            avg_val_loss = valid_loss / len(val_loader)
            avg_rmse = rmse_accum / len(val_loader)
            
            history['val_loss'].append(avg_val_loss)
            history['val_rmse'].append(avg_rmse)
            
            print(f' +++ Val Loss: {avg_val_loss:.3f} | Val RMSE: {avg_rmse:.3f} +++')

            # Save Best Model based on VALIDATION loss (Crucial!)
            if avg_val_loss < min_valid_loss:
                print(f"   ⭐ New Best Model! ({min_valid_loss:.4f} -> {avg_val_loss:.4f})")
                min_valid_loss = avg_val_loss
                torch.save(model.state_dict(), PATH_TO_BEST_MODEL)
        
        # Step Scheduler
        scheduler.step()

        # 5. Save Checkpoint
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_valid_loss': min_valid_loss,
            'history': history
        }, PATH_TO_LAST_CKPT)

        # 6. Save Logs
        with open(PATH_TO_LOG_FILE, 'w') as f:
            json.dump(history, f, indent=4)
                    
if __name__ == '__main__':
    train()