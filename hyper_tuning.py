import csv
import matplotlib.pyplot as plt
from datetime import datetime
from model import ThreeLayerNN
from train import train_model
from utils import save_model


def hyperparameter_search(X_train, y_train, X_val, y_val):
    search_space = {
        'hidden_sizes': [32, 64, 128, 256, 512, 1024],
        'learning_rates': [0.1, 0.05, 0.01, 0.005,0.001],
        'l2_lambdas': [0.1, 0.01, 0.005, 0.0001],
        'activations': ['relu', 'sigmoid']
    }
    
    best_acc = 0
    best_params = {}
    
    # 初始化日志文件
    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'hidden_size', 'learning_rate', 'l2_lambda', 
                        'activation', 'epoch', 'train_loss', 'val_loss', 'val_acc'])
    
    for hs in search_space['hidden_sizes']:
        for lr in search_space['learning_rates']:
            for reg in search_space['l2_lambdas']:
                for act in search_space['activations']:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    print(f"\n=== Training hs={hs}, lr={lr}, reg={reg}, act={act} ===")
                    
                    model = ThreeLayerNN(3072, hs, activation=act)
                    trained_model, history = train_model(
                        model, X_train, y_train, X_val, y_val,
                        lr=lr, l2_lambda=reg,
                        epochs=100, batch_size=512
                    )
                    
                    # 保存训练曲线
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(history['train_loss'], label='Train')
                    plt.plot(history['val_loss'], label='Validation')
                    plt.title('Loss Curve')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(history['val_acc'], label='Validation')
                    plt.title('Accuracy Curve')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.legend()
                    
                    filename = f"save_model/training_curve_best_val_acc={max(history['val_acc']):.4f}_hs{hs}_lr{lr}_reg{reg}_act{act}_{timestamp}.png"
                    plt.tight_layout()
                    plt.savefig(filename)
                    plt.close()
                    
                    # 记录日志
                    with open('training_log.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        for epoch_idx in range(len(history['train_loss'])):
                            row = [
                                timestamp,
                                hs,
                                lr,
                                reg,
                                act,
                                epoch_idx+1,
                                history['train_loss'][epoch_idx],
                                history['val_loss'][epoch_idx],
                                history['val_acc'][epoch_idx]
                            ]
                            writer.writerow(row)
                    
                    # 更新最佳参数
                    current_acc = max(history['val_acc'])
                    if current_acc > best_acc:
                        best_acc = current_acc
                        best_params = {
                            'hidden_size': hs,
                            'lr': lr,
                            'l2_lambda': reg,
                            'activation': act
                        }
                        print(f"New best accuracy: {best_acc:.4f}")
    
    print(f'\n=== Final Best Params ===\n{best_params}\nAccuracy: {best_acc:.4f}')
    return best_params


