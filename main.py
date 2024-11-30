import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.tcp import PcapDataset, split_dataset
from model.lstm import LSTMClassifier

if __name__ == "__main__":
    # 超参数

    seq_length = 100  # 序列长度
    batch_size = 8  # 批大小
    num_epochs = 10  # 训练轮数
    learning_rate = 0.001
    device = "cuda"

    # 创建数据集和 DataLoader
    dataset = PcapDataset()

    train_dataset, val_dataset = split_dataset(dataset, seed=2024)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # 初始化模型、损失函数和优化器
    model = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # 将数据传入设备
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("train finfish！")

    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)  # 将数据加载到设备
            outputs = model(data)  # 模型前向传播
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            total += labels.size(0)  # 累计样本数
            correct += (predicted == labels).sum().item()  # 累计正确预测数

    accuracy = correct / total  # 计算准确率
    print(f'acc: {accuracy}')