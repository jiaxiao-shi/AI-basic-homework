# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 加载MNIST简化版数据集（8x8图像）
digits = datasets.load_digits()    # 加载手写数字数据集
X, y = digits.data, digits.target  # X是特征，y是标签

# 2. 数据预处理：标准化数据
# 标准化特征（数据缩放至均值为0，方差为1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分数据集：训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. 创建并训练SVM模型
svm_classifier = SVC(kernel='linear')  # 使用线性核的SVM
svm_classifier.fit(X_train, y_train)

# 5. 预测测试集结果
y_pred = svm_classifier.predict(X_test)

# 6. 评估分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"分类准确率: {accuracy * 100:.2f}%")

# 7. 可视化结果
# 展示一些测试集的手写数字以及预测的标签
n_samples = 5  # 展示5个样本
fig, axes = plt.subplots(1, n_samples, figsize=(10, 3))

for i in range(n_samples):
    ax = axes[i]
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')  # 将数据转换成8x8的图像
    ax.set_title(f"Prediction: {y_pred[i]}", fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()
