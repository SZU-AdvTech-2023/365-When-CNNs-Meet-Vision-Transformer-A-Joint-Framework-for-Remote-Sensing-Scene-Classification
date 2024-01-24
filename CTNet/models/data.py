from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch


def Get_Data(test_split=0.2):  # 设置测试集比例为20%
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建ImageFolder实例
    dataset = datasets.ImageFolder(root='/Users/zero/Downloads/NWPU-RESISC45', transform=transform)

    # 计算测试集的大小
    test_size = int(test_split * len(dataset))
    train_size = int(len(dataset) - test_size)
    
    # 随机分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return train_dataset, test_dataset