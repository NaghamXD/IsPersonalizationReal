import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
PRETRAINED_PATH = "VSViG-base.pth"
DEVICE = torch.device("cpu") # CPU is fine for inspection

def inspect_first_layer():
    if not os.path.exists(PRETRAINED_PATH):
        print(f"❌ File not found: {PRETRAINED_PATH}")
        return

    print(f"🕵️‍♂️ Loading weights from {PRETRAINED_PATH}...")
    try:
        state_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    # 1. FIND THE FIRST LAYER
    # We look for the first key that has "weight" and corresponds to the visual backbone
    # usually named "backbone.0", "conv1", "patch_embed", or similar.
    
    first_layer_name = None
    first_layer_weights = None

    print("\n🔍 Scanning for the first visual layer...")
    for key, val in state_dict.items():
        # We look for a Conv2d weight with 3 input channels (RGB)
        # Shape is typically (Out_Channels, 3, Kernel_H, Kernel_W)
        if "weight" in key and val.ndim == 4 and val.shape[1] == 3:
            first_layer_name = key
            first_layer_weights = val
            break
    
    if first_layer_weights is None:
        print("❌ Could not identify the first Conv2d layer automatically.")
        # Fallback: Print first 5 keys to help user debug
        print("First 5 keys in state_dict:")
        for i, k in enumerate(state_dict.keys()):
            if i > 5: break
            print(f"  {k}  (Shape: {state_dict[k].shape})")
        return

    # 2. ANALYZE STATISTICS
    w = first_layer_weights.numpy()
    
    print(f"\n✅ FOUND First Layer: '{first_layer_name}'")
    print(f"   Shape: {w.shape} (Out, In=3, H, W)")
    
    mean_val = np.mean(w)
    std_val = np.std(w)
    abs_mean = np.mean(np.abs(w))
    max_val = np.max(w)
    min_val = np.min(w)

    print(f"\n📊 Weight Statistics:")
    print(f"   Mean:      {mean_val:.6f}")
    print(f"   Std Dev:   {std_val:.6f}")
    print(f"   Avg Mag:   {abs_mean:.6f}  (Average Absolute Value)")
    print(f"   Range:     [{min_val:.6f}, {max_val:.6f}]")

    # 3. INTERPRETATION
    print(f"\n💡 DIAGNOSIS:")
    
    if abs_mean < 0.001:
        print("   👉 Weights are TINY (< 0.001).")
        print("   ✅ The model likely expects RAW PIXELS (0-255).")
        print("   ACTION: Do NOT divide by 255.")
        
    elif abs_mean > 0.01:
        print("   👉 Weights are NORMAL (> 0.01).")
        print("   ✅ The model likely expects NORMALIZED inputs (0-1).")
        print("   ACTION: You MUST divide by 255.")
        
        # Check for Standardization hint (harder to be certain, but let's try)
        # If weights are distributed positively and negatively around 0, it supports Standardization.
        print("   ℹ️  Note: Standard ImageNet Normalization (Mean/Std) is highly likely for these weights.")

    else:
        print("   👉 Weights are ambiguous. Try both.")

    # 4. PLOT HISTOGRAM (Optional visualization)
    plt.figure(figsize=(8, 4))
    plt.hist(w.flatten(), bins=100, color='blue', alpha=0.7)
    plt.title(f"Weight Distribution of {first_layer_name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    inspect_first_layer()