import numpy as np

# 定义三层神经网络
class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, activation='relu'):
        """
        初始化三层神经网络的参数。

        参数:
        - input_size (int): 输入层的维度。
        - hidden_size (int): 隐藏层的神经元数量。
        - activation (str): 隐藏层的激活函数，可选 'relu' 或 'sigmoid'。
        """        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # 基于激活函数的初始化
        if activation == 'relu':
            # He初始化
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        else:  # sigmoid
            # Xavier初始化
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size)
            
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 10) * np.sqrt(1.0/hidden_size)
        self.b2 = np.zeros(10)
        
        # 前向传播缓存
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.probs = None
    
    def forward(self, X):
        """
        前向传播过程，计算输入数据 X 的输出概率分布。

        参数:
        - X (numpy.ndarray): 输入数据，形状为 (m, input_size)，其中 m 是样本数量。

        返回:
        - probs (numpy.ndarray): 输出层的概率分布，形状为 (m, 10)。
        """
        # 第一层
        self.z1 = X.dot(self.W1) + self.b1
        if self.activation == 'relu':
            self.a1 = np.maximum(0, self.z1)
        else:
            self.a1 = 1/(1 + np.exp(-self.z1))
        
        # 输出层
        self.z2 = self.a1.dot(self.W2) + self.b2
        
        # Softmax 输出概率分布
        exp_scores = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def compute_loss(self, y, l2_lambda):
        """
        计算带 L2 正则化的交叉熵损失。

        参数:
        - y (numpy.ndarray): 真实标签，形状为 (m,)。
        - l2_lambda (float): L2 正则化系数。

        返回:
        - loss (float): 总损失（数据损失 + 正则化损失）。
        """
        m = y.shape[0]
        log_probs = -np.log(self.probs[range(m), y] + 1e-8)
        data_loss = np.sum(log_probs)/m
        reg_loss = 0.5*l2_lambda*(np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss
    
    def backward(self, X, y, l2_lambda):
        """
        反向传播过程，计算梯度。

        参数:
        - X (numpy.ndarray): 输入数据，形状为 (m, input_size)。
        - y (numpy.ndarray): 真实标签，形状为 (m,)。
        - l2_lambda (float): L2 正则化系数。

        返回:
        - dW1 (numpy.ndarray): 隐藏层权重梯度。
        - db1 (numpy.ndarray): 隐藏层偏置梯度。
        - dW2 (numpy.ndarray): 输出层权重梯度。
        - db2 (numpy.ndarray): 输出层偏置梯度。
        """
        m = X.shape[0]
        
        # 将标签转换为 one-hot 编码
        y_one_hot = np.eye(10)[y]
        
        # 输出层梯度
        delta3 = self.probs - y_one_hot
        dW2 = (self.a1.T.dot(delta3))/m + l2_lambda*self.W2
        db2 = np.sum(delta3, axis=0)/m
        
        # 隐藏层梯度
        if self.activation == 'relu':
            delta2 = delta3.dot(self.W2.T) * (self.z1 > 0)
        else:
            delta2 = delta3.dot(self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = (X.T.dot(delta2))/m + l2_lambda*self.W1
        db1 = np.sum(delta2, axis=0)/m
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, lr):
        # 更新隐藏层权重
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
