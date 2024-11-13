import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import sys

# 1. 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 2. 配置 Matplotlib 使用指定的中文字体
def set_matplotlib_fonts():
    font_path = '/data/ljm/GestureRecognition/PaddleVideo/Key-point-sequence-action-recognition/simfang.ttf'  # 指定中文字体路径
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)  # 强制添加字体
        prop = font_manager.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = prop.get_name()
        print(f"已设置 Matplotlib 使用字体: {prop.get_name()}")
    else:
        print(f"字体文件未找到: {font_path}, 中文标签可能无法正常显示。")
        # 默认设置，中文可能无法显示
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Arial']

    # 打印当前字体配置
    print("当前 Matplotlib 使用的字体:", matplotlib.rcParams['font.family'])

set_matplotlib_fonts()

# 3. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 4. 定义自定义数据集类
class HandGestureDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): 所有样本的文件路径。
            labels (list): 每个样本的标签。
            transform (callable, optional): 可选的变换函数。
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx], self.labels[idx]
        data = pd.read_excel(file_path)

        # 获取所有帧ID，并排序
        frame_ids = sorted(data['frame_id'].unique())

        frames = len(frame_ids)
        num_landmarks = 21
        keypoints = np.zeros((frames, num_landmarks, 3), dtype=np.float32)  # (frames, landmarks, 3)

        for i, frame_id in enumerate(frame_ids):
            frame_data = data[(data['frame_id'] == frame_id) & (data['hand_id'] == 1)]  # 仅选择 hand_id=1
            if frame_data.empty:
                continue  # 如果该帧没有检测到指定 hand_id 的手，保持该帧的关键点为0

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

# 5. 定义数据加载与划分函数
def get_dataloaders(root_dir, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, hand_id=1):
    """
    使用分层采样（Stratified Sampling）进行数据集划分，确保每个子集包含所有类。
    """
    # 收集所有文件路径和对应标签
    file_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        for file_name in os.listdir(cls_dir):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(cls_dir, file_name)
                file_paths.append(file_path)
                labels.append(class_to_idx[cls])

    # 使用 sklearn 的 train_test_split 进行分层采样
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=test_ratio, random_state=42, stratify=labels
    )
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_ratio, random_state=42, stratify=y_temp
    )

    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")

    # 创建 DataLoader
    train_dataset = HandGestureDataset(X_train, y_train)
    val_dataset = HandGestureDataset(X_val, y_val)
    test_dataset = HandGestureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    return train_loader, val_loader, test_loader, class_names

# 6. 定义修正后的 ST-GCN 模型
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

# 7. 定义训练与验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_accuracy = 0
    trigger_times = 0

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

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
        history['train_loss'].append(epoch_loss)

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
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}, '
              f'LR: {current_lr:.6f}')

        # 学习率调度
        scheduler.step()

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
    return history

# 8. 定义测试与评估函数
def evaluate_model(model, test_loader, class_names, num_classes):
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
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵
    plt.close()

    # 打印分类报告
    print("分类报告：")
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=list(range(num_classes))))

# 9. 可视化训练过程
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # 创建保存图像的目录
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.figure(figsize=(14,5))

    # 绘制损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制学习率
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lr'], 'bo-', label='学习率')
    plt.title('学习率变化')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./training_history.png')  # 保存训练历史
    plt.close()

# 10. 主函数
def main():
    # 数据集根目录
    root_dir = r'/data/ljm/GestureRecognition/PaddleVideo/Key-point-sequence-action-recognition'

    # 获取 DataLoader，调整比例为 80% 训练，10% 验证，10% 测试
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root_dir=root_dir,
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        hand_id=1
    )

    # 定义模型
    num_classes = len(class_names)
    model = ST_GCN(in_channels=3, num_classes=num_classes).to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型，并记录训练历史
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        patience=10
    )

    # 可视化训练过程
    plot_training_history(history)

    # 评估模型
    evaluate_model(model, test_loader, class_names, num_classes)

if __name__ == "__main__":
    main()