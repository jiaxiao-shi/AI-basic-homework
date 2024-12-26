from sklearn.linear_model import LogisticRegression
 
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split
from matplotlib import pyplot as plt

# print(1) 
 
# 获取数据集数据和标签
datas = load_digits()
X_data = datas.data
y_data = datas.target
 
#  展示前十个数据的图像
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )
ax = ax.flatten()
for i in range(10):
    ax[i].imshow(datas.data[i].reshape((8, 8)), cmap='Greys', interpolation='nearest')
plt.show()
 
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
# 建立逻辑回归模型
model = LogisticRegression(max_iter=10000, random_state=42, multi_class='multinomial')
scores = cross_val_score(model, X_train, y_train, cv=10)  # 十折交叉验证
k = 0
for i in scores:
    k += i
print("十折交叉验证平均值：", k / 10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error_rate = model.score(X_test, y_test)
 
print(f"十折交叉验证:{scores}\n")
print(f"错误率:{error_rate}\n")
print(f"测试集预测值:{y_pred}\n")