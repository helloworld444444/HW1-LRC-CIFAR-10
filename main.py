from utils import load_cifar10
from model import ThreeLayerNN
from train import train_model
from hyper_tuning import hyperparameter_search
from test import test_model
import numpy as np

if __name__ == '__main__':
    # 数据加载
    
    data_dir = 'cifar-10-batches-py'
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(data_dir)
    
    # 超参数搜索
    best_params = hyperparameter_search(X_train, y_train, X_val, y_val)
    
    # 使用最佳参数训练最终模型
    final_model = ThreeLayerNN(
        3072, 
        best_params['hidden_size'],
        activation=best_params['activation']
    )
    
    # 合并数据集
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    # 最终训练
    trained_model, _ = train_model(
        final_model, X_full, y_full,
        X_test, y_test,
        lr=best_params['lr'],
        l2_lambda=best_params['l2_lambda'],
        epochs=100,
        batch_size=512,
        final=True,
    )
    
    # 最终评估
    test_acc = test_model(trained_model, X_test, y_test)