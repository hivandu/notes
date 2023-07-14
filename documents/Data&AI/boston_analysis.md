# Boston house analysis

> The source code: [Boston House](https://github.com/hivandu/colab/blob/master/AI_data/boston_house.ipynb)

```python
# Import package
# Used to load the Boston housing price data set
from sklearn.datasets import load_boston
# pandas toolkit If you are unfamiliar with pandas, you can refer to the official 10-minute tutorial: https://pandas.pydata.org/pandas-docs/stable/10min.html
import pandas as pd
import numpy as np
# seaborn for drawing
import seaborn as sns
import matplotlib.pyplot as plt
# Show drawing
%matplotlib inline


data = load_boston() # load datase

data.keys() # Fields inside data

df = pd.DataFrame(data['data'])

# Looking at the first 5 rows of the dataframe, we can see that the column names are numbers
df.head(5)

data['feature_names'] # Feature name
```

## The Table params and chinese info


```
params	chinese info
CRIM	住房所在城镇的人均犯罪率
ZN	    住房用地超过 25000 平方尺的比例
INDUS	住房所在城镇非零售商用土地的比例
CHAS	有关查理斯河的虚拟变量（如果住房位于河边则为1,否则为0 ）
NOX	    一氧化氮浓度
RM	    每处住房的平均房间数
AGE	    建于 1940 年之前的业主自住房比例
DIS	    住房距离波士顿五大中心区域的加权距离
RAD	    离住房最近的公路入口编号
TAX     每 10000 美元的全额财产税金额
PTRATIO	住房所在城镇的师生比例
B	    1000(Bk-0.63)^2,其中 Bk 指代城镇中黑人的比例
LSTAT	弱势群体人口所占比例
MEDV	业主自住房的中位数房价（以千美元计）
```

```python
# Replace numeric column names with feature names
df.columns = data['feature_names']
df.head(5)

# The target is the house price, which is also our target value. We assign the target value to the dataframe
df['price'] = data['target']
df.head(5)

# View the correlation coefficient between the feature and price, positive correlation and negative correlation
sns.heatmap(df.corr(), annot=True, fmt='.1f')

plt.scatter(df['RM'], df['price'])


plt.figure(figsize=(20, 5))

# View the data distribution display of some features and price
features = ['LSTAT', 'RM']
target = df['price']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker = 'o')
    plt.title('{} price'.format(col))
    plt.xlabel(col)
    plt.ylabel('price')


# Simple example: univariate forecast price
x = df['RM']
y = df['price']

history_notes = {_x: _y for _x, _y in zip(x,y)}

history_notes[6.575]


# Find the top three prices that are closest to RM:6.57,
similary_ys = [y for _, y in sorted(history_notes.items(), key=lambda x_y: (x_y[0] - 6.57) ** 2)[:3]]
similary_ys


# Calculate the average of three
np.mean(similary_ys)
```


### Use historical data to predict data that has never been seen before, the most direct method

## K-Neighbor-Nearst

```python
def knn(query_x, history, top_n = 3):
    sorted_notes = sorted(history.items(), key = lambda x_y: (x_y[0] - query_x)**2) 
    similar_notes = sorted_notes[:top_n]
    similar_ys = [y for _, y in similar_notes]

    return np.mean(similar_ys)

knn(5.4, history_notes)
```

In order to obtain results faster, we hope to obtain predictive power by fitting a function

$$ f(rm) = k * rm + b $$ 

## Random Approach

$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} (\hat{y_i} - y_i) ^ 2 $$
$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} ((k * rm_i + b) - y_i) ^ 2 $$

```python
def loss(y_hat, y):
    return np.mean((y_hat - y)**2)

import random

min_loss = float('inf')

best_k, best_b = None, None


for step in range(1000):
    min_v, max_v = -100, 100
    k, b = random.randrange(min_v, max_v), random.randrange(min_v, max_v)
    y_hats = [k * rm_i + b for rm_i in x]
    current_loss = loss(y_hats, y)

    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print(f'{step}, we have func f(rm) = {k} * rm + {b},  lss is :{current_loss}')

plt.scatter(x, y)
plt.scatter(x, [best_k * rm + best_b for rm in x])
```

## Monte Carlo simulation(蒙特卡洛模拟)

### Supervisor

$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} ((k * rm_i + b) - y_i) ^ 2 $$

$$ \frac{\partial{loss(k, b)}}{\partial{k}} = \frac{2}{n}\sum_{i \in N}(k * rm_i + b - y_i) * rm_i $$

$$ \frac{\partial{loss(k, b)}}{\partial{b}} = \frac{2}{n}\sum_{i \in N}(k * rm_i + b - y_i)$$


```python
def partial_k(k, b, x, y):
    return 2  * np.mean((k*x+b-y) * x)

def partial_b(k, b, x, y):
    return 2 * np.mean(k*x+b-y)

k, b = random.random(), random.random()
min_loss = float('inf')

best_k, best_b = None, None
learning_rate = 1e-2

for step in range(2000):
    k, b = k + (-1 * partial_k(k, b, x, y) * learning_rate), b + (-1 * partial_b(k, b, x, y) * learning_rate)
    y_hats = k * x + b
    current_loss = loss(y_hats, y)

    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print(f'setp {step}, we have func f(rm) = {k} * rm + {b},  lss is :{current_loss}')

best_k, best_b


plt.scatter(x, y)
plt.scatter(x, [best_k * rm + best_b for rm in x])
```

## Supervised Learning

We turn the forecast of housing prices into a more responsible and sophisticated model. What should we do?

$$ f(x) = k * x + b $$

$$ f(x) = k2 * \sigma(k_1 * x + b_1) + b2 $$

$$ \sigma(x) = \frac{1}{1 + e^(-x)} $$ 

```python
def sigmoid(x):
    return 1 / (1+np.exp(-x))

sub_x = np.linspace(-10, 10)
plt.plot(sub_x, sigmoid(sub_x))


def random_linear(x):
    k, b = random.random(), random.random()
    return k * x + b

def complex_function(x):
    return (random_linear(x))

for _ in range(10):
    index = random.randrange(0, len(sub_x))
    sub_x_1, sub_x_2 = sub_x[:index], sub_x[index:]
    new_y = np.concatenate((complex_function(sub_x_1), complex_function(sub_x_2)))
    plt.plot(sub_x, new_y)
```

We can implement more complex functions through simple, basic modules and repeated superposition

For more and more complex functions? How does the computer seek guidance?

1. What is machine learning?
2. The shortcomings of this method of KNN, what is the background of the proposed linear fitting
3. How to get faster function weight update through supervision method
4. The combination of nonlinear and linear functions can fit very complex functions
5. Deep learning we can fit more complex functions through basic function modules

### Assigment:

$$ L2-Loss(y, \hat{y}) = \frac{1}{n}\sum{(\hat{y} - y)}^2 $$

$$ L1-Loss(y, \hat{y}) = \frac{1}{n}\sum{|(\hat{y} - y)|} $$


L2-Loss becomes L1Loss and achieves gradient descent

Realize L1Loss gradient descent from 0

#### 1. import package

```python
import numpy as np
import pandas as pd
```

#### 2. load data

```python
from sklearn.datasets import load_boston
data = load_boston()
data.keys()

data_train = data.data
data_traget = data.target

df = pd.DataFrame(data_train, columns = data.feature_names)
df.head()

df.describe() # Data description, you can view the statistics of each variable
```
#### 3. Data preprocessing

Normalization or standardization can prevent a certain dimension or a few dimensions from affecting the data too much when there are very many dimensions, and secondly, the program can run faster. There are many methods, such as standardization, min-max, z-score, p-norm, etc. How to use it depends on the characteristics of the data set.

[Further reading-数据标准化的迷思之深度学习领域](https://zhuanlan.zhihu.com/p/81560511)

```python
from sklearn.preprocessing import StandardScaler
# z = (x-u) / s u is the mean, s is the standard deviation
ss = StandardScaler() 
data_train = ss.fit_transform(data_train)
# For linear models, normalization or standardization is generally required, otherwise gradient explosion will occur, and tree models are generally not required
data_train = pd.DataFrame(data_train, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
data_train.describe() 

# y=Σwixi+
# Because the derivation of b is all 1, add a bias b to the data and set it to 1, as a feature of the data and update the gradient wi*b=wi
data_train['bias']  = 1
data_train

```

Divide the data set, where 20% of the data is used as the test set X_test, y_test, and the other 80% are used as the training set X_train, y_train, where random_state is the random seed

```python
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_train, data_traget, test_size = 0.2, random_state=42)

print('train_x.shape, train_y.shape', train_x.shape, train_y.shape)
print('test_x.shape, test_y.shape', test_x.shape, test_y.shape)

train_x = np.array(train_x)
```

### Model training and gradient update

```python
def l1_cost(x, y, theta):
    """
    x: 特征
    y: 目标值
    thta: 模型参数
    """
    k = x.shape[0]
    total_cost = 0
    for i in range(k):
        total_cost += 1/k * np.abs(y[i] -theta.dot(x[i, :]))
    return total_cost

def l2_cost(x, y, theta):
    k = x.shape[0]
    total_cost = 0
    for i in range(k):
        total_cost += 1/k * (y[i] -theta.dot(x[i,:])) ** 2
    return total_cost

np.zeros(10).shape

def step_l1_gradient(x, y, learning_rate, theta):
    """
    Function to calculate the gradient of the MAE loss function
    Return the gradient value 0 for the non-differentiable point at 0
    X:特征向量
    y：目标值
    learing_rate:学习率
    theta:参数
    """
    n = x.shape[0]
        # print(n)
    e = y - x @ theta
    gradients = - (x.T @ np.sign(e)) / n # sign is a sign function
    thata = theta - learning_rate * gradients
    return theta

def step_l2_gradient(x, y, learning_rate, theta):
    k = x.shape[0]
    n = x.shape[1]
    gradients = np.zeros(n)
    for i in range(k):
        for j in range(n):
            gradients[j] += (-2/k) * (y[i] - (theta.dot(x[i, :]))) * x[i, j]
    theta = theta - learning_rate * gradient
    return theta

# def step_gradient(X, y, learning_rate, theta):
#     """
#     X:特征向量
#     y：目标值
#     learing_rate:学习率
#     theta:参数
#     """
#     m_deriv = 0
#     N = len(X)
#     for i in range(N):
#         # 计算偏导
#         # -x(y - (mx + b)) / |mx + b|
#         m_deriv += - X[i] * (y[i] - (theta*X[i] + b)) / abs(y[i] - (theta*X[i] + b))
#     # We subtract because the derivatives point in direction of steepest ascent
#     theta -= (m_deriv / float(N)) * learning_rate
# #     theta = theta - learning_rate * gradients
#     return theta

def gradient_descent(train_x, train_y, learning_rate, iterations):
    k = train_x.shape[0]
    n = train_x.shape[1]
    theta = np.zeros(n) # Initialization parameters

    loss_values = []
    # print(theta.shape)

    for i in range(iterations):
        theta = step_l1_gradient(train_x, train_y, learning_rate, theta)
        loss = l1_cost(train_x, train_y, theta)
        loss_values.append(loss)
        print(i, 'cost:', loss)
    return theta, loss_values

#  Training parameters
learning_rate = 0.04 # Learning rate
iterations = 300 # Number of iterations
theta, loss_values = gradient_descent(train_x, train_y, learning_rate, iterations)
```