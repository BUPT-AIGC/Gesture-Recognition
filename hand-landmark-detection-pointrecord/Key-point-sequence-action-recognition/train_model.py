import os
# 1. 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 定义自定义数据集类
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None, hand_id=1):
        """
        Args:
            root_dir (str): 数据集根目录，每个子文件夹代表一个动作类别。
            transform (callable, optional): 可选的变换函数。
            hand_id (int): 选择的手ID，这里为1。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.hand_id = hand_id
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        print(f"找到 {len(self.samples)} 个样本，{len(self.classes)} 个类别。")

    def _make_dataset(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for file_name in os.listdir(cls_dir):
                if file_name.endswith('.xlsx'):
                    file_path = os.path.join(cls_dir, file_name)
                    samples.append((file_path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = pd.read_excel(file_path)

        # 获取所有帧ID，并排序
        frame_ids = sorted(data['frame_id'].unique())

        frames = len(frame_ids)
        num_landmarks = 21
        keypoints = np.zeros((frames, num_landmarks, 3), dtype=np.float32)  # (frames, landmarks, 3)

        for i, frame_id in enumerate(frame_ids):
            frame_data = data[(data['frame_id'] == frame_id) & (data['hand_id'] == self.hand_id)]
            if frame_data.empty:
                continue  # 如果该帧没有检测到指定hand_id的手，保持该帧的关键点为0

            for _, row in frame_data.iterrows():
                landmark_id = int(row['landmark_id'])
                x = row['x']
                y = row['y']
                z = row['z']
                keypoints[i, landmark_id] = [x, y, z]

        if self.transform:
            keypoints = self.transform(keypoints)

        # 转换为Tensor，并调整维度为 (channels, frames, landmarks)
        keypoints = torch.from_numpy(keypoints).permute(2, 0, 1)  # (channels, frames, landmarks)
        label = torch.tensor(label, dtype=torch.long)

        return keypoints, label

# 2. 定义数据加载与划分
def get_dataloaders(root_dir, batch_size=32, val_split=0.2, test_split=0.1, hand_id=1):
    dataset = HandGestureDataset(root_dir=root_dir, hand_id=hand_id)
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, dataset.classes

# 3. 定义修正后的 ST-GCN 模型
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, residual=True):
        super(ST_GCN_Block, self).__init__()
        padding = (kernel_size // 2, 0)  # 在时间维度上进行填充

        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))  # 空间卷积
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                             stride=(stride,1), padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if residual:
            if (in_channels == out_channels) and (stride ==1):
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.residual = None

    def forward(self, x):
        out = self.gcn(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.tcn(out)
        out = self.bn2(out)

        if self.residual is not None:
            res = self.residual(x)
            out += res

        out = self.relu(out)
        return out

class ST_GCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ST_GCN, self).__init__()
        self.layer1 = ST_GCN_Block(in_channels, 64, kernel_size=3, stride=1, residual=False)
        self.layer2 = ST_GCN_Block(64, 128, kernel_size=3, stride=2, residual=True)
        self.layer3 = ST_GCN_Block(128, 256, kernel_size=3, stride=2, residual=True)
        self.layer4 = ST_GCN_Block(256, 512, kernel_size=3, stride=2, residual=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels=3, frames, landmarks=21)
        x = self.layer1(x)  # -> (batch_size, 64, frames, landmarks)
        x = self.layer2(x)  # -> (batch_size, 128, frames/2, landmarks)
        x = self.layer3(x)  # -> (batch_size, 256, frames/4, landmarks)
        x = self.layer4(x)  # -> (batch_size, 512, frames/8, landmarks)
        x = self.global_pool(x)  # -> (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)  # -> (batch_size, 512)
        x = self.fc(x)  # -> (batch_size, num_classes)
        return x

# 4. 定义训练与验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_accuracy = 0
    trigger_times = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)  # (batch_size, channels, frames, landmarks)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)
                outputs = model(val_data)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item() * val_data.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == val_labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}')

        # 检查是否是最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_st_gcn_model.pth')
            print("保存最佳模型")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"早停计数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("早停触发，停止训练")
                break

    print("训练完成。最佳验证准确率: {:.4f}".format(best_accuracy))

# 5. 定义测试与评估函数
def evaluate_model(model, test_loader, class_names):
    model.load_state_dict(torch.load('best_st_gcn_model.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)
            outputs = model(test_data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

    # 打印分类报告
    print("分类报告：")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# 6. 主函数
def main():
    # 数据集根目录
    root_dir = r'/data/ljm/GestureRecognition/PaddleVideo/Key-point-sequence-action-recognition'

    # 获取DataLoader
    train_loader, val_loader, test_loader, class_names = get_dataloaders(root_dir=root_dir, batch_size=32, val_split=0.2, test_split=0.1, hand_id=1)

    # 定义模型
    num_classes = len(class_names)
    model = ST_GCN(in_channels=3, num_classes=num_classes).to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10)

    # 评估模型
    evaluate_model(model, test_loader, class_names)

if __name__ == "__main__":
    main()