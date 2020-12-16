# 波士顿房价CART回归树



## 上代码



```python
# CART回归树预测
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor,export_graphviz
import graphviz

# 准备数据集
boston = load_boston()

# 探索数据
print(boston.feature_names)

# 获取特征集和房价
features = boston.data
prices = boston.target


# 随机抽取33%的数据作为测试集，其余作为训练集
train_features, test_features, train_price, test_price = train_test_split(features,prices,test_size=0.33)

# 创建CART回归树
dtr = DecisionTreeRegressor()

# 拟合构造CART回归树
dtr.fit(train_features, train_price)

# 预测测试集中的房价
predict_price = dtr.predict(test_features)

grap_data = export_graphviz(dtr, out_file=None)
graph = graphviz.Source(grap_data)

# 测试集的结果评价
print(f'回归树二乘偏差均值:',mean_squared_error(test_price, predict_price))
print(f'回归树绝对值偏差均值:',mean_absolute_error(test_price, predict_price))

# 生成回归树可视化
graph.render('Boston')
```



!> 跑这段代码之前请先保证已安装好相关依赖；