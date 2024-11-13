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

    def normalize_keypoints(self, keypoints):
        # 简单归一化到 [0, 1] 范围，基于数据集的最大最小值
        min_vals = keypoints.min(axis=(0,1), keepdims=True)
        max_vals = keypoints.max(axis=(0,1), keepdims=True)
        keypoints = (keypoints - min_vals) / (max_vals - min_vals + 1e-6)
        return keypoints

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

        keypoints = self.normalize_keypoints(keypoints)

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

# 6. 定义邻接矩阵的创建和归一化函数
def create_adjacency_matrix(connections, num_landmarks):
    A = torch.zeros(num_landmarks, num_landmarks)
    for i, j in connections:
        A[i, j] = 1
        A[j, i] = 1  # 无向图
    # 添加自连接
    A += torch.eye(num_landmarks)
    return A

def normalize_adjacency_matrix(A):
    D = torch.diag(torch.sum(A, dim=1))
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0  # 处理除以零
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

# 7. 定义包含图卷积的 ST-GCN 模型
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=3, stride=1, residual=True):
        super(ST_GCN_Block, self).__init__()
        padding = (kernel_size // 2, 0)  # 仅在时间维度上进行填充

        # 空间图卷积，使用邻接矩阵
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 时间卷积
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                             stride=(stride, 1), padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        if residual:
            if (in_channels == out_channels) and (stride == 1):
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.residual = None

        # 注册邻接矩阵为缓冲区（不作为模型参数）
        self.register_buffer('A', A)

    def forward(self, x):
        # x 的形状: (batch_size, in_channels, frames, landmarks)

        # 图卷积
        out = self.gcn(x)  # (batch_size, out_channels, frames, landmarks)
        # 转置为 (batch_size, frames, out_channels, landmarks) 以便进行矩阵乘法
        out = out.permute(0, 2, 1, 3)  # (batch, frames, channels, landmarks)
        out = torch.matmul(out, self.A.T)  # (batch, frames, channels, landmarks)
        out = out.permute(0, 2, 1, 3)  # (batch, channels, frames, landmarks)
        out = self.bn1(out)
        out = self.relu(out)

        # 时间卷积
        out = self.tcn(out)
        out = self.bn2(out)

        # 残差连接
        if self.residual is not None:
            res = self.residual(x)
            out += res

        out = self.relu(out)
        return out

class ST_GCN(nn.Module):
    def __init__(self, in_channels, num_classes, A):
        super(ST_GCN, self).__init__()
        self.layer1 = ST_GCN_Block(in_channels, 64, A, kernel_size=3, stride=1, residual=False)
        self.layer2 = ST_GCN_Block(64, 128, A, kernel_size=3, stride=2, residual=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.fc = nn.Linear(128, num_classes)  # 调整全连接层

    def forward(self, x):
        # x 的形状: (batch_size, channels=3, frames, landmarks=21)
        x = self.layer1(x)  # -> (batch_size, 64, frames, landmarks)
        x = self.layer2(x)  # -> (batch_size, 128, frames/2, landmarks)
        x = self.global_pool(x)  # -> (batch_size, 128, 1, 1)
        x = x.view(x.size(0), -1)  # -> (batch_size, 128)
        x = self.dropout(x)        # 应用 Dropout
        x = self.fc(x)             # -> (batch_size, num_classes)
        return x

# 8. 定义训练与验证函数（修改后的版本）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, models_dir='models', logs_dir='logs'):
    best_train_loss = float('inf')  # 初始化最佳训练损失
    best_val_loss = float('inf')    # 初始化最佳验证损失
    best_val_accuracy = 0           # 初始化最佳验证准确率

    best_train_loss_epoch = -1
    best_val_loss_epoch = -1
    best_val_accuracy_epoch = -1

    trigger_times = 0

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # 记录训练过程
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': [],
        'best_train_loss': [],
        'best_val_loss': [],
        'best_val_accuracy': [],
        'save_best_train_loss': [],
        'save_best_val_loss': [],
        'save_best_val_accuracy': []
    }

    # 确保模型和日志保存目录存在
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        history['epoch'].append(epoch)

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

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

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

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        # 学习率调度
        scheduler.step(epoch_val_accuracy)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        print(f'Epoch [{epoch}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Acc: {epoch_val_accuracy:.4f}, '
              f'LR: {current_lr:.6f}')

        # 初始化保存标记
        save_best_train_loss = False
        save_best_val_loss = False
        save_best_val_accuracy = False

        # 检查是否是最佳训练损失
        if epoch_train_loss <= best_train_loss:
            best_train_loss = epoch_train_loss
            best_train_loss_epoch = epoch
            best_train_loss_path = os.path.join(models_dir, 'best_train_loss_model.pth')
            torch.save(model.state_dict(), best_train_loss_path)  # 保存最佳训练损失模型
            print(f"保存训练集损失最低的模型: {best_train_loss_path} (Epoch {epoch})")
            save_best_train_loss = True

        # 检查是否是最佳验证损失
        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_loss_epoch = epoch
            best_val_loss_path = os.path.join(models_dir, 'best_val_loss_model.pth')
            torch.save(model.state_dict(), best_val_loss_path)  # 保存最佳验证损失模型
            print(f"保存验证集损失最低的模型: {best_val_loss_path} (Epoch {epoch})")
            save_best_val_loss = True

        # 检查是否是最佳验证准确率
        if epoch_val_accuracy >= best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_val_accuracy_epoch = epoch
            best_val_accuracy_path = os.path.join(models_dir, 'best_val_accuracy_model.pth')
            torch.save(model.state_dict(), best_val_accuracy_path)  # 保存最佳验证准确率模型
            print(f"保存验证集精度最高的模型: {best_val_accuracy_path} (Epoch {epoch})")
            save_best_val_accuracy = True
            trigger_times = 0  # 重置早停计数
        else:
            trigger_times += 1
            print(f"早停计数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("早停触发，停止训练")
                break

        # 记录保存标记和对应的轮次
        history['best_train_loss'].append(best_train_loss)
        history['best_val_loss'].append(best_val_loss)
        history['best_val_accuracy'].append(best_val_accuracy)
        history['save_best_train_loss'].append(save_best_train_loss)
        history['save_best_val_loss'].append(save_best_val_loss)
        history['save_best_val_accuracy'].append(save_best_val_accuracy)

    # 保存最后一轮的模型
    last_epoch_path = os.path.join(models_dir, 'last_epoch_model.pth')
    torch.save(model.state_dict(), last_epoch_path)
    print(f"保存最后一轮的模型: {last_epoch_path}")

    # 在 history 中记录最后一轮
    history['best_train_loss'].append(best_train_loss)
    history['best_val_loss'].append(best_val_loss)
    history['best_val_accuracy'].append(best_val_accuracy)
    history['save_best_train_loss'].append(False)
    history['save_best_val_loss'].append(False)
    history['save_best_val_accuracy'].append(False)

    print("训练完成。")
    print(f"最佳训练集损失: {best_train_loss:.4f} (Epoch {best_train_loss_epoch})")
    print(f"最低验证集损失: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
    print(f"最高验证集准确率: {best_val_accuracy:.4f} (Epoch {best_val_accuracy_epoch})")

    # 记录训练日志到 CSV 文件
    logs_path = os.path.join(logs_dir, 'training_log.csv')
    df_history = pd.DataFrame(history)
    df_history.to_csv(logs_path, index=False)
    print(f"训练日志已保存到 {logs_path}")

    return history

# 9. 定义测试与评估函数（修改后的版本）
def evaluate_model(model, test_loader, class_names, num_classes, cm_save_path='confusion_matrix.png'):
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
    # 调整横轴标签角度，避免截断
    plt.xticks(rotation=45, ha='right')  # 旋转标签为45度，并且右对齐
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    # 调整布局，确保标签不会被截断
    plt.tight_layout()
    plt.savefig(cm_save_path)  # 保存混淆矩阵
    plt.close()

    # 打印分类报告
    print(f"分类报告 ({os.path.basename(cm_save_path)}):")
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=list(range(num_classes))))

# 10. 定义训练过程可视化函数
def plot_training_history(history, save_path='training_history.png'):
    epochs = history['epoch']

    plt.figure(figsize=(18,5))

    # 绘制训练与验证损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失')
    plt.title('训练与验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制验证准确率
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='验证准确率')
    plt.title('验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制学习率变化
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['lr'], 'bo-', label='学习率')
    plt.title('学习率变化')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # 保存训练历史
    plt.close()
    print(f"训练历史已保存到 {save_path}")

# 11. 主函数（修改后的版本）
def main():
    # 定义保存目录
    root_dir = '/data/ljm/GestureRecognition/PaddleVideo/Key-point-sequence-action-recognition/dataset'  # 数据集根目录
    models_dir = 'models'  # 模型保存目录
    confusion_matrices_dir = 'confusion_matrices'  # 混淆矩阵保存目录
    logs_dir = 'logs'  # 日志保存目录

    # 创建保存目录
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(confusion_matrices_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 获取 DataLoader，调整比例为 80% 训练，10% 验证，10% 测试
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        root_dir=root_dir,
        batch_size=32,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        hand_id=1
    )

    # 定义关键点之间的连接关系
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]

    num_landmarks = 21

    # 创建并归一化邻接矩阵
    A = create_adjacency_matrix(connections, num_landmarks)  # (21, 21)
    A_norm = normalize_adjacency_matrix(A)  # (21, 21)

    # 将邻接矩阵移动到设备（GPU 或 CPU）
    A_norm = A_norm.to(device)

    # 定义模型
    num_classes = len(class_names)
    model = ST_GCN(in_channels=3, num_classes=num_classes, A=A_norm).to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减

    # 训练模型，并记录训练历史
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=20,
        models_dir=models_dir,
        logs_dir=logs_dir
    )

    # 可视化训练过程
    training_history_path = os.path.join(logs_dir, 'training_history.png')
    plot_training_history(history, save_path=training_history_path)

    # 定义要评估的模型及其描述
    models_to_evaluate = {
        '训练集损失最低模型': os.path.join(models_dir, 'best_train_loss_model.pth'),
        '验证集损失最低模型': os.path.join(models_dir, 'best_val_loss_model.pth'),
        '验证集精度最高模型': os.path.join(models_dir, 'best_val_accuracy_model.pth'),
        '最后一轮模型': os.path.join(models_dir, 'last_epoch_model.pth')
    }

    # 遍历并评估每个模型
    for description, model_path in models_to_evaluate.items():
        if not os.path.exists(model_path):
            print(f"模型文件未找到: {model_path}, 跳过评估 {description}。")
            continue

        print(f"\n评估 {description} ({model_path})")

        # 加载模型权重
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        # 定义混淆矩阵的保存路径
        safe_description = description.replace('/', '_').replace('\\', '_')  # 处理描述中可能的特殊字符
        cm_save_path = os.path.join(confusion_matrices_dir, f'confusion_matrix_{safe_description}.png')

        # 评估模型并保存混淆矩阵
        evaluate_model(model, test_loader, class_names, num_classes, cm_save_path=cm_save_path)

    print("\n所有模型的评估已完成。")

if __name__ == "__main__":
    main()