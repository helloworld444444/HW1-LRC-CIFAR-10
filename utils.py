import numpy as np
import pickle
import os



# 数据加载与预处理 (改进点1：修复数据泄露)
def load_cifar10(data_dir):
    """加载CIFAR-10数据集并进行标准化预处理"""
    train_files = [f'data_batch_{i}' for i in range(1,6)]
    test_file = 'test_batch'
    
    # 加载训练数据
    X_train = []
    y_train = []
    for file in train_files:
        with open(os.path.join(data_dir, file), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X_train.append(data[b'data'])
            y_train.extend(data[b'labels'])
    
    # 分割训练集和验证集
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    X_val = X_train[40000:]
    y_val = y_train[40000:]
    X_train = X_train[:40000]
    y_train = y_train[:40000]
    
    # 加载测试数据
    with open(os.path.join(data_dir, test_file), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X_test = data[b'data']
        y_test = np.array(data[b'labels'])
    
    # 数据预处理（统一使用训练集的统计量）
    X_train = X_train.astype(np.float32) / 255.0
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # 应用相同的标准化到所有数据集
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val.astype(np.float32)/255 - mean) / (std + 1e-8)
    X_test = (X_test.astype(np.float32)/255 - mean) / (std + 1e-8)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_model(model, filename):
    """保存模型参数"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)


def load_model(model, filename):
    """加载模型参数"""
    params = np.load(filename)
    model.W1 = params['W1']
    model.b1 = params['b1']
    model.W2 = params['W2']
    model.b2 = params['b2']
    return model




