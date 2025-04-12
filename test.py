import numpy as np

def evaluate_model(model, X_test, y_test):
    """模型评估函数"""
    probs = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y_test)

def test_model(model, X_test, y_test):
    """完整的测试流程"""
    acc = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
    return acc