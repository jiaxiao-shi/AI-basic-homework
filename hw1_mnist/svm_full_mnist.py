# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 加载MNIST数据集
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")

# 获取数据和标签
X = mnist.data                # 图像数据
y = mnist.target.astype(int)  # 数字标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 输出数据集的形状和一些样本数据
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 创建SVM分类器并训练模型
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# 保存模型
joblib.dump(svm, 'svm_full_mnist_model.pkl')

# 在测试集上进行预测
y_pred = svm.predict(X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM分类器的准确率: {accuracy * 100:.2f}%")

# 可视化一些测试集样本及其预测结果
n_samples = 20
n_col = 5
n_row = n_samples // n_col
fig, axes = plt.subplots(n_row, n_col, figsize=(10, 7))
axes = axes.ravel()
for i in range(n_samples):
    axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"Prediction: {y_pred[i]}")
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
plt.show()
