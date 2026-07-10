'''import json
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_PATH = "checkpoints/training_log.json"

def plot_history():
    if not os.path.exists(LOG_PATH):
        print(f"File {LOG_PATH} not found. Train the model first.")
        return

    with open(LOG_PATH, 'r') as f:
        history = json.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    # Create X-axis steps
    epochs = range(1, len(train_loss) + 1)
    # Validation runs every 5 epochs (based on your train.py logic)
    val_epochs = [i * 5 for i in range(1, len(val_loss) + 1)]

    plt.figure(figsize=(10, 5))
    
    # Plot Training Loss
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.6)
    
    # Plot Validation Loss
    if val_loss:
        plt.plot(val_epochs, val_loss, label='Validation Loss', color='red', marker='o')
        
        # Find best epoch
        min_val_loss = min(val_loss)
        best_val_idx = val_loss.index(min_val_loss)
        best_epoch = val_epochs[best_val_idx]
        
        plt.title(f"Train vs Val Loss (Best Epoch: {best_epoch}, Loss: {min_val_loss:.4f})")
        
        # Highlight best point
        plt.scatter(best_epoch, min_val_loss, s=100, c='green', zorder=5, label='Best Model')
        plt.annotate(f'Best: {min_val_loss:.3f}', (best_epoch, min_val_loss), xytext=(0, 10), textcoords='offset points')
    else:
        plt.title("Training Loss (No Validation Data Yet)")

    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_history()'''

import json
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_PATH = "checkpoints_improved/training_log.json"

def plot_history():
    if not os.path.exists(LOG_PATH):
        print(f"File {LOG_PATH} not found. Train the model first.")
        return

    with open(LOG_PATH, 'r') as f:
        history = json.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    val_rmse = history.get('val_rmse', [])  # <--- NEW: Load RMSE history
    
    # Create X-axis steps
    epochs = range(1, len(train_loss) + 1)
    # Validation runs every 5 epochs (based on your train.py logic)
    val_epochs = [i * 5 for i in range(1, len(val_loss) + 1)]

    plt.figure(figsize=(12, 6))  # Slightly larger figure for clarity
    
    # Plot Training Loss
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.6)
    
    # Plot Validation Loss
    if val_loss:
        plt.plot(val_epochs, val_loss, label='Validation Loss', color='red', marker='o')
        
        # 1. Find the index of the Minimum Validation Loss
        min_val_loss = min(val_loss)
        best_val_idx = val_loss.index(min_val_loss)
        best_epoch = val_epochs[best_val_idx]
        
        # 2. Get the corresponding RMSE for that specific epoch
        # (We assume val_rmse has the same length as val_loss)
        rmse_info = ""
        annotation_text = f'Loss: {min_val_loss:.4f}'
        
        if val_rmse and len(val_rmse) > best_val_idx:
            best_rmse = val_rmse[best_val_idx]
            rmse_info = f", RMSE: {best_rmse:.2f}%"
            annotation_text += f'\nRMSE: {best_rmse:.2f}%'

        # 3. Update Title to show both metrics
        plt.title(f"Train vs Val Loss - Improved (Best Epoch: {best_epoch}, Loss: {min_val_loss:.4f}{rmse_info})", fontsize=12, fontweight='bold')
        
        # 4. Highlight best point with green dot
        plt.scatter(best_epoch, min_val_loss, s=150, c='green', zorder=5, label='Best Model')
        
        # 5. Add text annotation box pointing to the best epoch
        plt.annotate(annotation_text, 
                     (best_epoch, min_val_loss), 
                     xytext=(0, 20), 
                     textcoords='offset points', 
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="green", lw=1.5))
    else:
        plt.title("Training Loss (No Validation Data Yet)")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_history()