# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ImprovedCNN, RobustResNet
from datareader import get_data_loaders

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    # Pindahkan model ke device yang sesuai
    model = model.to(device)
    
    # Definisikan loss function dan optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Untuk menyimpan history training
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print("Memulai training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val Accuracy: {accuracy:.2f}%')
        
        # Simpan model terbaik
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model terbaik disimpan dengan akurasi: {accuracy:.2f}%')
        
        print('-' * 60)
    
    return train_losses, val_losses, val_accuracies

def plot_training_history(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    # Set random seed untuk reproducibility
    torch.manual_seed(42)
    
    # Tentukan device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    BATCH_SIZE = 32
    train_loader, val_loader, n_classes, n_channels = get_data_loaders(BATCH_SIZE)
    
    # Inisialisasi model (pilih salah satu)
    # model = ImprovedCNN(in_channels=n_channels, num_classes=n_classes)
    model = RobustResNet(in_channels=n_channels, num_classes=n_classes)
    
    # Training
    train_losses, val_losses, val_accuracies = train_model(
        model, 
        train_loader, 
        val_loader,
        num_epochs=50,
        device=device
    )
    
    # Plot hasil training
    plot_training_history(train_losses, val_losses, val_accuracies)
