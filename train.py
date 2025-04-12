import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from utils import save_model



def train_model(model, X_train, y_train, X_val, y_val,
               lr, l2_lambda, 
               epochs, batch_size,
               lr_decay=0.95, decay_every=5, final=False):
    lr_init = lr
    best_val_acc = 0.0
    best_params = None
    n_samples = X_train.shape[0]
    history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # 学习率衰减
        if (epoch+1) % decay_every == 0:
            lr *= lr_decay
            print(f"Learning rate decay to {lr:.4f}")
        
        # 随机打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_train_loss = 0.0
        num_batches = 0
        
        # Mini-batch训练
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播
            model.forward(X_batch)
            # 计算损失
            loss = model.compute_loss(y_batch, l2_lambda)
            total_train_loss += loss
            num_batches += 1
            
            # 反向传播
            dW1, db1, dW2, db2 = model.backward(X_batch, y_batch, l2_lambda)
            # 参数更新
            model.update_params(dW1, db1, dW2, db2, lr)
        
        # 记录训练损失
        avg_train_loss = total_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # 验证集评估
        val_probs = model.forward(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val)
        val_loss = model.compute_loss(y_val, l2_lambda)
        
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} => '
              f'Train loss: {avg_train_loss:.4f}, '
              f'Val loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                'W1': model.W1.copy(),
                'b1': model.b1.copy(),
                'W2': model.W2.copy(),
                'b2': model.b2.copy()
            }

    # 恢复并保存最佳参数
    if best_params:
        model.W1 = best_params['W1']
        model.b1 = best_params['b1']
        model.W2 = best_params['W2']
        model.b2 = best_params['b2']
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if final==False:
        np.savez(f'save_model/best_model_best_val_acc={best_val_acc:.4f}_hs={model.hidden_size}_act={model.activation}_lr={lr_init}_l2_lambda={l2_lambda}_epochs={epochs}_batch_size={batch_size}+{timestamp}.npz', **best_params)  
    else:
        np.savez(f'save_model/best_model_test_acc={best_val_acc:.4f}_hs={model.hidden_size}_act={model.activation}_lr={lr_init}_l2_lambda={l2_lambda}_epochs={epochs}_batch_size={batch_size}+{timestamp}.npz', **best_params)  

        
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
                    
        filename = f"save_model/finalmodel_acc={max(history['val_acc']):.4f}_hs={model.hidden_size}_act={model.activation}_lr={lr_init}_l2_lambda={l2_lambda}_epochs={epochs}_batch_size={batch_size}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()    
    return model, history
