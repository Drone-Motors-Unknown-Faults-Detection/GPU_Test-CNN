import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from datetime import datetime
from tqdm import tqdm

# 設為 False 可關閉訓練紀錄匯出
ENABLE_LOGGING = True


class TrainingLogger:
    """將訓練過程紀錄並匯出成 txt 檔。enabled=False 時所有方法皆為 no-op。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.epoch_records = []
        self.start_time: datetime = datetime.now()
        self.total_time: float = 0.0

        # 取得裝置資訊，供匯出時寫入
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.device_info = f"{self.device_name} ({total_mem:.2f} GB)"
        else:
            self.device_type = "cpu"
            self.device_name = "CPU"
            self.device_info = "CPU"

    def start(self):
        """記錄訓練開始時間，應在訓練迴圈前呼叫。"""
        if not self.enabled:
            return
        self.start_time = datetime.now()

    def log_epoch(self, epoch: int, total_epochs: int, loss: float, train_acc: float, test_acc: float, elapsed: float):
        """記錄單一 epoch 的結果，應在每個 epoch 結束後呼叫。"""
        if not self.enabled:
            return
        self.epoch_records.append({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "elapsed": elapsed,
        })

    def finish(self, total_time: float):
        """記錄總訓練時間，應在訓練迴圈結束後呼叫。"""
        if not self.enabled:
            return
        self.total_time = total_time

    def export(self, output_dir: str = "."):
        """將完整訓練紀錄寫入 txt，檔名包含訓練開始時間。"""
        if not self.enabled:
            return
        os.makedirs(output_dir, exist_ok=True)
        filename = self.start_time.strftime("training_log_%Y%m%d_%H%M%S.txt")
        filepath = os.path.join(output_dir, filename)

        lines = []
        lines.append("=" * 50)
        lines.append("MNIST CNN 訓練紀錄")
        lines.append("=" * 50)
        lines.append(f"訓練裝置類型 : {self.device_type.upper()}")
        lines.append(f"裝置名稱     : {self.device_info}")
        lines.append(f"訓練開始時間 : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"總訓練時間   : {self.total_time:.2f} 秒")
        lines.append("")
        lines.append("-" * 50)
        lines.append(f"{'Epoch':<8} {'Loss':<10} {'Train Acc':<12} {'Test Acc':<12} {'累計時間'}")
        lines.append("-" * 50)
        for r in self.epoch_records:
            lines.append(
                f"{r['epoch']}/{r['total_epochs']:<5} "
                f"{r['loss']:<10.4f} "
                f"{r['train_acc']:<12.2f} "
                f"{r['test_acc']:<12.2f} "
                f"{r['elapsed']:.2f}s"
            )
        lines.append("-" * 50)
        lines.append("")
        lines.append("訓練完成")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"訓練紀錄已儲存至 {filepath}")


class CNN(nn.Module):
    """兩層卷積 + 兩層全連接的 CNN，用於 MNIST 10 類分類。"""

    def __init__(self):
        super(CNN, self).__init__()
        # 卷積層：1 通道輸入 → 32 特徵圖，padding=2 保持 28×28 尺寸
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # 卷積層：32 → 64 特徵圖
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # 每次 Pooling 將尺寸減半：28→14→7
        self.pool = nn.MaxPool2d(2, 2)
        # 攤平後 64×7×7 = 3136 個特徵
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # 訓練時隨機關閉 50% 神經元，防止過擬合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32×28×28 → 32×14×14
        x = self.pool(self.relu(self.conv2(x)))  # 64×14×14 → 64×7×7
        x = x.view(x.size(0), -1)               # 攤平成 (batch, 3136)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                          # 輸出 10 個 logits
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """執行一個 epoch 的訓練，回傳平均 loss 與訓練準確率。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="訓練"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 清除上一步殘留的梯度，再反向傳播更新權重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def test(model, test_loader, device):
    """在測試集上評估模型，回傳準確率。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # 推論不需要計算梯度，節省記憶體
        for images, labels in tqdm(test_loader, desc="測試"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # Windows 多行程需要在 __main__ 保護下啟動，否則 DataLoader worker 會重複執行整個腳本
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
        print(f"GPU可用記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    logger = TrainingLogger(enabled=ENABLE_LOGGING)

    batch_size = 128
    learning_rate = 0.001
    epochs = 10

    # 標準化參數來自 MNIST 全資料集的均值與標準差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("\n加載 MNIST 數據...")
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # pin_memory=True 讓 CPU→GPU 資料傳輸更快；num_workers 使用多行程預載資料
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n開始訓練...\n")
    start_time = time.time()
    logger.start()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        logger.log_epoch(epoch + 1, epochs, train_loss, train_acc, test_acc, elapsed)

    total_time = time.time() - start_time
    print(f"\n訓練完成！耗時: {total_time:.2f} 秒")

    logger.finish(total_time)
    logger.export(output_dir="./logs")

    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("模型已保存為 mnist_cnn.pth")