# GPU-MNIST

用 CNN / ResNet 模型訓練影像分類，包含 MNIST 手寫數字與 CIFAR-10 彩色圖片，並說明如何讓訓練跑在 GPU 上。

---

## MNIST（手寫數字辨識）

**腳本**：`src/app.py`

### 資料集

- 訓練集：60,000 張 28×28 灰階圖
- 測試集：10,000 張
- 類別：數字 0 ~ 9，共 10 類

### 模型架構

兩層卷積 + 兩層全連接的簡單 CNN：

```
輸入 (1×28×28)
→ Conv2d(1→32) + ReLU + MaxPool  →  32×14×14
→ Conv2d(32→64) + ReLU + MaxPool →  64×7×7
→ Flatten → FC(3136→128) + ReLU + Dropout(0.5)
→ FC(128→10)  →  10 個類別
```

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 128 |
| Learning Rate | 0.001 |
| Epochs | 10 |
| Optimizer | Adam |

### 執行

```bash
python src/app.py
```

訓練紀錄輸出至 `logs/MNIST/`

---

## CIFAR-10（彩色圖片分類）

**腳本**：`src/Cifar10.py`

### 資料集

- 訓練集：50,000 張 32×32 彩色圖（RGB 3 通道）
- 測試集：10,000 張
- 類別：飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車，共 10 類

### 模型架構

簡化版 ResNet，三組殘差層逐步縮小特徵圖：

```
輸入 (3×32×32)
→ Conv2d(3→64) + BN + ReLU
→ Layer1: 2× ResidualBlock(64→64)   →  64×32×32
→ Layer2: 2× ResidualBlock(64→128)  →  128×16×16
→ Layer3: 2× ResidualBlock(128→256) →  256×8×8
→ AdaptiveAvgPool → Flatten
→ FC(256→10)  →  10 個類別
```

ResidualBlock 的 shortcut 連接讓梯度可以跳層傳遞，解決深層網路難以訓練的問題。

### 超參數

| 項目 | 值 |
|------|----|
| Batch Size | 256 |
| Learning Rate | 0.001（每 10 epoch × 0.5） |
| Epochs | 20 |
| Optimizer | Adam |
| LR Scheduler | StepLR(step=10, gamma=0.5) |

訓練集使用隨機裁切（RandomCrop）與水平翻轉（RandomHorizontalFlip）做資料增強。

### 執行

```bash
python src/Cifar10.py
```

訓練紀錄輸出至 `logs/CIFAR10/`

---

## 如何讓模型跑在 GPU 上

PyTorch 用一行就能偵測 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

有 NVIDIA GPU 且安裝了 CUDA 驅動時，`device` 會是 `cuda`，否則自動退回 `cpu`。

接著只要把**模型**和**資料**都搬到同一個 device 上即可：

```python
model = ResNet().to(device)   # 模型搬到 GPU

images = images.to(device)   # 每個 batch 的資料也要搬
labels = labels.to(device)
```

這樣前向傳播、反向傳播、梯度更新全部都會在 GPU 上執行。

### DataLoader 的 pin_memory

```python
DataLoader(..., num_workers=4, pin_memory=True)
```

- `pin_memory=True`：將資料預先鎖定在記憶體，讓 CPU → GPU 的傳輸更快
- `num_workers=4`：用多個子行程預先載入資料，避免 GPU 等待

> **Windows 注意事項**：`num_workers > 0` 在 Windows 上需要把訓練程式包在 `if __name__ == '__main__':` 裡，否則會因為 spawn 機制重複執行腳本而出錯。

---

## 訓練紀錄

每次訓練結束後自動將結果匯出為 txt，依資料集分資料夾存放：

```
logs/
├── MNIST/
│   └── training_log_20260502_001118.txt
└── CIFAR10/
    └── training_log_20260502_012345.txt
```

透過頂部的 `ENABLE_LOGGING` 旗標控制是否匯出：

```python
ENABLE_LOGGING = True   # 改為 False 可關閉
```

---

## 環境需求

- Python 3.8+
- PyTorch（建議安裝 CUDA 版本）
- torchvision
- tqdm

```bash
pip install torch torchvision tqdm
```

CUDA 版 PyTorch 請至 [pytorch.org](https://pytorch.org) 依照作業系統與 CUDA 版本選擇對應的安裝指令。
