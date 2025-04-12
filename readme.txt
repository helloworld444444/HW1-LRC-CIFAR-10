CIFAR-10图像分类项目（三层神经网络实现）

== 项目概述 ==
基于NumPy实现的三层神经网络，用于CIFAR-10数据集的图像分类，包含：
数据预处理流水线
可调整的神经网络架构
超参数调优模块
带验证的训练流程
测试集完整评估

== 文件结构 ==
├── utils.py - 数据加载/预处理 & 模型保存加载
├── model.py - 三层神经网络类定义
├── train.py - 训练循环实现
├── hyper_tuning.py- 超参数网格搜索
├── test.py - 模型评估函数
├── main.py - 主执行脚本
├── hyper_selected - 直接使用最佳超参数训练和测试
└── save_model/ - 模型参数保存目录

== 核心功能 ==
• 数据处理：
规范化的数据分割 (40000/10000/10000)
标准化处理（使用训练集统计量）防止数据泄露
超参数搜索时40000张训练->10000张验证
寻找到最优参数后40000+10000张训练->10000张测试

• 模型架构：
输入层(3072) → 隐藏层(可配置) → 输出层(10类别)
激活函数选项：ReLU/Sigmoid
He/Xavier初始化策略
支持L2正则化

• 训练特性：
小批量梯度下降
学习率衰减（每5个epoch衰减15%）
基于验证准确率的早停机制
最佳模型检查点保存

• 超参数搜索：
网格搜索范围：
→ 隐藏层维度[32, 64, 128, 256, 512, 1024]
→ 学习率[0.1, 0.05, 0.01, 0.005,0.001]
→ L2正则化强度[0.1, 0.01, 0.005, 0.0001]
→ 隐藏层激活函数类型['relu', 'sigmoid']

• 自动记录CSV训练日志

• 生成训练曲线可视化图表

• 测试评估：
提供evaluate_model单次评估
完整测试流程test_model封装

== 环境要求 ==
Python 3.6+

必需依赖库：
numpy
matplotlib
pickle
os
csv
datetime

== 快速开始 ==
准备数据：
确保CIFAR-10数据位于项目目录的cifar-10-batches-py文件夹

安装依赖：
pip install numpy matplotlib

运行完整流程：
python main.py

直接选择最优超参数查看效果：
python hyper_selected.py

== 输出文件 ==
• save_model/ 包含：
最佳模型参数(.npz文件)
训练曲线图(.png)

• training_log.csv - 完整的超参数搜索记录

== 输出 ==
[训练阶段]
Epoch 38/100 => Train loss: 4.8994, Val loss: 4.9164, Val Acc: 0.3935
Epoch 39/100 => Train loss: 4.8634, Val loss: 4.8820, Val Acc: 0.3939
Learning rate decay to 0.0066
Epoch 40/100 => Train loss: 4.8300, Val loss: 4.8489, Val Acc: 0.3926
Epoch 41/100 => Train loss: 4.7985, Val loss: 4.8165, Val Acc: 0.3937

在save_model文件夹下产生training_curve+hyper-parameters+timestamp.png存储每组超参数训练曲线
在save_model文件夹下产生best_model_best_val_acc+hyper-parameters+timestamp.npz储存每组超参数训练得到的最优参数


[测试阶段]
典型最佳超参数 ：隐藏层维度: 512| 初始学习率: 0.01|L2强度: 0.01 | 激活函数: ReLU
在save_model文件夹下产生best_model_test_acc=0.5410_hs=512_act=relu_lr=0.01_l2_lambda=0.01_epochs=100_batch_size=512+20250412125527.npz储存最优超参数训练得到的最优参数
在save_model文件夹下产生finalmodel_acc=0.5410_hs=512_act=relu_lr=0.01_l2_lambda=0.01_epochs=100_batch_size=512_20250412125527.png储存最优超参数训练的过程
最优超参数下的Test Accuracy: 0.5410

== 实现细节 ==
数值稳定性：
Softmax计算时减去最大值防止数值溢出
在除法和log运算中添加eps(1e-8)防止除0错误

正则化策略：
L2惩罚同时作用于两个全连接层
正则化项包含在损失和梯度计算中

优化策略：
每个epoch随机打乱数据

== 注意事项 ==
超参数寻找每轮训练100个epoch，完整超参数搜索共尝试6*5*4*2=240个组合，在CPU上训练约需48小时
最终模型训练训练100个epoch，需约15分钟，hyper_selected运行产生的val_acc本质上即为最终模型的test_acc
不同随机种子可能导致结果微小差异
超参数搜索寻找的最优参数和training_curve文件较大，仅保留效果较好的training_curve图片和最终最佳训练产生的模型参数
reg和l2_lambda在代码中均表示正则项






