import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ImprovedCNN, RobustResNet
from datareader import get_data_loaders

def train_model(model, train_loader, val_loader, num_epochs=60, device='cuda'):
    model = model.to(device)

    # --- Hitung rasio kelas untuk pos_weight ---
    class_counts = torch.zeros(2)
    for _, labels in train_loader.dataset:
        class_counts[int(labels.item())] += 1
    pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
    pos_weight = torch.tensor([pos_weight], device=device)
    print(f"[INFO] Pos weight digunakan: {pos_weight.item():.4f}")

    # --- Loss function dan optimizer ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)

    # --- Scheduler kombinasi ---
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=4)

    # --- History ---
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # --- Gradient clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === VALIDATION ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # --- Scheduler update ---
        scheduler_cosine.step()
        scheduler_plateau.step(avg_val_loss)

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- Simpan model terbaik (berdasarkan validasi) ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Model terbaik disimpan! (Akurasi: {val_acc:.2f}%)")

    return train_losses, val_losses, val_accs

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Hyperparameter tuned for better validation ===
    BATCH_SIZE = 64
    EPOCHS = 60

    train_loader, val_loader, n_classes, n_channels = get_data_loaders(BATCH_SIZE)

    model = RobustResNet(in_channels=n_channels, num_classes=n_classes, use_pretrained=False)
    # model = ImprovedCNN(in_channels=n_channels, num_classes=n_classes)

    train_losses, val_losses, val_accs = train_model(model, train_loader, val_loader, num_epochs=EPOCHS, device=device)

    # --- Plot history ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses,label='Train Loss'); plt.plot(val_losses,label='Val Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(val_accs,label='Val Acc'); plt.legend()
    plt.tight_layout(); plt.savefig('training_history.png'); plt.show()
