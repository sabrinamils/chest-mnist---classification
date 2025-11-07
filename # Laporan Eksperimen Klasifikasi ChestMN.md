# Laporan Eksperimen Klasifikasi ChestMNIST
![Header](header.png)

## Anggota Kelompok
1. Nindi Gustina Putri (122430044)
2. Sabrina Milesa (122439105)
3. Tata Widyawasih (122430012)

## Ringkasan Proyek
Proyek ini mengimplementasikan klasifikasi biner pada dataset ChestMNIST untuk membedakan antara kasus Cardiomegaly dan Pneumothorax menggunakan deep learning.

## Komponen Utama

### 1. Data Processing (datareader.py)
- Menggunakan dataset ChestMNIST dari medmnist
- Memfilter data untuk klasifikasi biner:
  - Kelas 0: Cardiomegaly
  - Kelas 1: Pneumothorax
- Implementasi data augmentation dan normalisasi
- Pemisahan dataset menjadi training dan validation set

### 2. Model Architecture (model.py)
Mengimplementasikan dua arsitektur model:

#### a. ImprovedCNN
- Arsitektur CNN sederhana dengan peningkatan bertahap
- Fitur utama:
  - Conv-BN-ReLU blocks
  - Progressive channel growth (16→32→64)
  - Dropout (p=0.5)
  - Batch Normalization
  - AdaptiveAvgPool2d

#### b. RobustResNet
- Berbasis arsitektur ResNet18
- Modifikasi untuk single-channel input
- Fitur utama:
  - Transfer learning option
  - Adaptive input channels
  - Dropout (p=0.4)
  - Custom classifier head

### 3. Training Process (train.py)
- Optimizer: AdamW dengan weight decay
- Learning rate scheduling dengan ReduceLROnPlateau
- Binary Cross Entropy loss
- Validation monitoring dan model checkpointing
- Training visualization dengan progress bars

## Hasil Eksperimen

### Dataset Statistics
```
Training Set:
- Cardiomegaly: [jumlah sampel]
- Pneumothorax: [jumlah sampel]

Validation Set:
- Cardiomegaly: [jumlah sampel]
- Pneumothorax: [jumlah sampel]
```

### Model Performance
```
Model: [ImprovedCNN/RobustResNet]
Best Validation Accuracy: [X.XX]%
Training Time: [XX] minutes
```

## Visualisasi
- Training/validation loss curves tersimpan di 'training_history.png'
- Contoh prediksi model pada validation set

## Kesimpulan & Pembelajaran
1. [Insight tentang performa model]
2. [Tantangan yang dihadapi]
3. [Potential improvements]

## Setup Instructions

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install medmnist matplotlib numpy tqdm
```

3. Run training:
```bash
python train.py
```

## References
1. MedMNIST Dataset: https://medmnist.com/
2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
3. ResNet Paper: "Deep Residual Learning for Image Recognition"

---
*Last Updated: November 7, 2025*