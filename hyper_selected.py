from utils import load_cifar10
from model import ThreeLayerNN
from train import train_model
import numpy as np
from test import test_model

# 选择的最佳超参数字典
# 包含隐藏层大小、学习率、L2正则化系数和激活函数类型
selected = {'hs':512,'lr':0.01,'reg':0.01,'act':'relu'}

# 数据集路径
data_dir = 'cifar-10-batches-py'

# 加载 CIFAR-10 数据集
# 返回训练集、验证集和测试集的特征和标签
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(data_dir)

# 创建最终的三层神经网络模型
# 输入维度为 3072（CIFAR-10 图像的扁平化维度）
# 隐藏层大小和激活函数根据最佳超参数字典设置
final_model = ThreeLayerNN(
        3072, 
        hidden_size=selected['hs'],
        activation=selected['act']
    )

# 将训练集和验证集合并为一个完整的训练集
# 用于最终模型的训练
X_full = np.concatenate([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

# 训练最终模型
# 使用完整的训练集和测试集
# 设置学习率、L2正则化系数、训练轮数、批量大小等超参数
trained_model, _ = train_model(
    final_model, X_full, y_full,
    X_test, y_test,
    lr=selected['lr'],
    l2_lambda=selected['reg'],
    epochs=100,
    batch_size=512,
    final=True,
)
# 测试最终模型的准确率，使用测试集评估模型性能
# 实际上，这里产生的val_acc即直接为test_acc
test_acc = test_model(trained_model, X_test, y_test)

