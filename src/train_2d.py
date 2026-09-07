"""
train_2d.py
Training engine for physical-layer RF spectrum classification:
- Memory-optimized PyTorch DataLoader loading cached `stft_dataset.npz` with `pin_memory=True`.
- AdamW optimizer (weight decay 1e-4) & ReduceLROnPlateau learning rate scheduler.
- Early Stopping saving optimal model weights to `checkpoints/best_model.pth`.
- Detailed per-epoch metrics and per-SNR validation breakdown.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from models_2d import get_model, STFT_RADN


class STFTDataset(Dataset):
    """PyTorch Dataset wrapping memory-cached STFT numpy arrays [N, 1, 32, 33]."""
    def __init__(self, npz_path="data/stft_dataset.npz"):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"STFT dataset not found at {npz_path}. Run 02_stft_converter.py first.")
        
        data = np.load(npz_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)
        self.snrs = torch.tensor(data['snrs'], dtype=torch.float32)
        self.class_names = list(data['class_names'])
        
        # Enforce tensor shape validation
        assert self.X.shape[1:] == (1, 32, 33), f"Dataset tensor shape mismatch! Got {self.X.shape[1:]}"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snrs[idx]


def evaluate_snr_breakdown(model, test_loader, device, class_names):
    """Evaluates classification accuracy across distinct SNR levels (-20 dB to 0 dB)."""
    model.eval()
    snr_correct = {}
    snr_total = {}
    
    with torch.no_grad():
        for inputs, targets, snrs in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            for pred, target, snr in zip(preds, targets, snrs):
                snr_val = float(snr.item())
                if snr_val not in snr_correct:
                    snr_correct[snr_val] = 0
                    snr_total[snr_val] = 0
                snr_total[snr_val] += 1
                if pred == target:
                    snr_correct[snr_val] += 1
                    
    print("\n" + "="*50)
    print("      SNR (dB) CLASSIFICATION ACCURACY BREAKDOWN")
    print("="*50)
    sorted_snrs = sorted(snr_correct.keys())
    for snr_val in sorted_snrs:
        acc = (snr_correct[snr_val] / snr_total[snr_val]) * 100.0
        bar = "█" * int(acc / 5)
        print(f" SNR: {snr_val:5.1f} dB | Accuracy: {acc:6.2f}% | {bar}")
    print("="*50 + "\n")


def train_model(
    npz_path="data/stft_dataset.npz",
    model_name="stft_radn",
    epochs=15,
    batch_size=64,
    lr=1e-3,
    checkpoint_dir="checkpoints",
    patience=5
):
    """Main training loop."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[INFO] Using compute device: {device}")
    
    # Load dataset
    full_dataset = STFTDataset(npz_path)
    num_samples = len(full_dataset)
    train_len = int(0.70 * num_samples)
    val_len = int(0.15 * num_samples)
    test_len = num_samples - train_len - val_len
    
    train_set, val_set, test_set = random_split(
        full_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Memory-optimized DataLoaders with pin_memory=True
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    print(f"[INFO] Dataset splits: Train={train_len}, Val={val_len}, Test={test_len}")
    
    # Initialize Model
    model = get_model(model_name, num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: {model_name.upper()} | Trainable Params: {total_params:,}")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n[START] Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        
        scheduler.step(epoch_val_loss)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpoint Saving & Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'model_name': model_name,
                'class_names': full_dataset.class_names
            }, best_ckpt_path)
            print(f"  [CHECKPOINT] Saved best weights to {best_ckpt_path} (Val Loss: {epoch_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Triggered after {patience} epochs without validation loss improvement.")
                break
                
    # Load best model for evaluation
    print(f"\n[INFO] Loading optimal checkpoint weights from {best_ckpt_path}...")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation & per-SNR breakdown
    evaluate_snr_breakdown(model, test_loader, device, full_dataset.class_names)
    print(f"[SUCCESS] Training complete. Optimal Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train_model(epochs=12, batch_size=64)
