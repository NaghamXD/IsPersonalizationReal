import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.vsvig import VSViG_base, VSViG_light
from src.data.dataset import VSViGDataset
from torch.utils.data import DataLoader
import torch, json
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

# --- CONFIGURATION ---
PROCESSED_FOLDER = "processed_data"
PATH_TO_DATA_FOLDER = PROCESSED_FOLDER

# --- PATH CONFIGS ---
CHECKPOINT_DIR = "outputs/standard/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PATH_TO_BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PATH_TO_LAST_CKPT  = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
PATH_TO_LOG_FILE   = os.path.join(CHECKPOINT_DIR, "training_log.json")

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
    dataset_train = VSViGDataset(data_folder=PATH_TO_DATA_FOLDER, label_file=train_label_path)
    dataset_val   = VSViGDataset(data_folder=PATH_TO_DATA_FOLDER, label_file=val_label_path)

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