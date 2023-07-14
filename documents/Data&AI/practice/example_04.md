# Machine Learning Part-02

- Data 
- Pre-processing 
- Feature-Extractor
- Split Training, Test, Validation
- Build Model
- Gradient Descent 
- Evaluation
- Predicat
- Analysis

House Price Regression

```python
## load data
from sklearn.datasets import load_boston
## ususlly will load in csv
data = load_boston()
print(data['DESCR'])
"""
_boston_dataset:

Boston house prices dataset
---------------------------

**Data Set Characteristics:**  

    :Number of Instances: 506 
show more (open the raw output data in a text editor) ...

Morgan Kaufmann.
"""
import pandas as pd
import numpy as np
df = pd.DataFrame(data['data'])
df.columns = data['feature_names']
df[df['CHAS'] == 1]
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
142	3.32105	0.0	19.58	1.0	0.8710	5.403	100.0	1.3216	5.0	403.0	14.7	396.90	26.82
...
	1.1296	24.0	666.0	20.2	347.88	8.88
"""
## Pre-processing
df.std()
"""
CRIM         8.601545
ZN          23.322453
INDUS        6.860353
CHAS         0.253994
NOX          0.115878
RM           0.702617
AGE         28.148861
DIS          2.105710
RAD          8.707259
TAX        168.537116
PTRATIO      2.164946
B           91.294864
LSTAT        7.141062
dtype: float64
"""

df['CHAS'] = df['CHAS'].astype('int')
df['CHAS'] = df['CHAS'].astype('category')
df['RAD'] = df['RAD'].astype('int')
df['RAD'] = df['RAD'].astype('category')

df
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296.0	15.3	396.90	4.98
...
505	0.04741	0.0	11.93	0	0.573	6.030	80.8	2.5050	1	273.0	21.0	396.90	7.88
506 rows × 13 columns
"""

df['RAD']
"""
0      1
1      2
2      2
...
505    1
Name: RAD, Length: 506, dtype: category
Categories (9, int64): [1, 2, 3, 4, ..., 6, 7, 8, 24]
"""

from sklearn.preprocessing import OneHotEncoder
onehoter = OneHotEncoder()
chas_and_rad_vec = onehoter.fit_transform(df[['CHAS', 'RAD']])

## Standarlize
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df.shape
"""
(506, 13)
"""

real_vec = ss.fit_transform(df.drop(columns = ['CHAS', 'RAD']))
chas_and_rad_vec[0].toarray()
"""
array([[1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]])
"""

import numpy as np
np.mean(real_vec, axis = 0)
"""
array([-1.12338772e-16,  7.89881994e-17,  2.10635198e-16, -1.96592852e-16,
       -1.08828186e-16, -1.47444639e-16, -8.42540793e-17,  0.00000000e+00,
       -4.21270397e-16, -7.44244367e-16, -3.08931624e-16])
"""

np.std(real_vec, axis = 0)
"""
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
"""

real_vec.shape
"""
(506, 11)
"""

chas_and_rad_vec.shape
"""
(506, 11)
"""

## Feature-Extractor
X = np.concatenate((real_vec, chas_and_rad_vec.toarray()), axis = 1)

y = data['target']

## Split Training, Test, Validation
def split_train_val_test(X, y, test_ratio = 0.2, val_ratio = 0.2):
    indices = np.random.choice(range(len(X)), size = len(X), replace=False)
    train_indices = indices[:int(len(X) * (1-test_ratio) * (1 - val_ratio))]
    val_indices = indices[int(len(X)*(1-test_ratio) * (1-val_ratio)): int(len(X) * (1-test_ratio))]
    test_indices = indices[int(len(X) * (1-test_ratio)):]

    return (X[train_indices], y[train_indices]), (X[val_indices], y[val_indices]), (X[test_indices], y[test_indices])

(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y)
```

> sklearn.model_selection.train_test_split also could be used

#### Build-Model

```python
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train, y_train)
"""
LinearRegression()
"""
```

> Question: If overfittiing or underfitting? 
> Explain: Why validation set is more useful in deep learning

#### Gradient Descent

#### Evaluation

```python
regression.score(X_train, y_train)
"""
0.7477980609064946
"""

regression.score(X_val, y_val)
"""
0.7611715890963341
"""

regression.score(X_test, y_test)
"""
0.711869928554872
"""

## Interpreter
regression.coef_
"""
array([-1.04208922,  1.30263494,  0.29143618, -2.31827512,  2.40383155,
        0.25013857, -3.55953868, -1.68823412, -2.37743843,  0.74411049,
       -3.79489254, -0.79143926,  0.79143926, -2.51995654, -2.20671004,
        0.65594998, -0.31683083, -0.07929752, -2.15244627, -0.06686364,
        1.93167854,  4.75447632])
"""

regression.intercept_
"""
22.070279554739386
"""

### Predict
X_test[0]
"""
array([ 1.68404594, -0.48772236,  1.01599907,  1.07378711,  0.21279502,
        1.11749449, -0.93188642,  1.53092646,  0.80657583, -3.61192313,
        2.29842066,  1.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  1.        ])
"""

regression.predict([X_test[0]])
"""
array([9.64589284])
"""

import matplotlib.pyplot as plt
for i in range(5):
    plt.scatter(X[:, 5], y)
    plt.scatter(X[:, 5], regression.predict(X))

plt.show()

```

![image-20210831230756185](http://qiniu.hivan.me/picGo/20210831230756.png?imgNote)



```python
import matplotlib
matplotlib.colors
%matplotlib inline

def show_predication_result(x, target):
    width = 3

    fig,ax = plt.subplots(x.shape[1]//width + 1, width, figsize = (40,40))

    for i in range(x.shape[1]):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        ax[ix].title.set_text('Feature-{}'.format(i))
        plt.scatter(x[:, i], target)
        plt.scatter(x[:, i], regression.predict(x))
        
show_predication_result(X_train, y_train)
```

![image-20210831230855270](http://qiniu.hivan.me/picGo/20210831230855.png?imgNote)

```python
show_predication_result(X_val, y_val)
```

![image-20210831230921925](http://qiniu.hivan.me/picGo/20210831230921.png?imgNote)



```python
show_predication_result(X_test, y_test)
```

![image-20210831230945492](http://qiniu.hivan.me/picGo/20210831230945.png?imgNote)



#### Outliers

### Part-02 Logstic Regression

- Data 
- Pre-processing 
- Feature-Extractor
- Split Training, Test, Validation
- Build Model
- Gradient Descent 
- Evaluation
- Predicat
- Analysis

Pre-processing

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from struct import unpack

def loadmnist(imagefile, labelfile):

    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Get metadata for images
    images.read(4) # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]

    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    # Get data
    x = np.zeros((N, rows*cols), dtype = np.uint8) #Initialize numpy array
    y = np.zeros(N, dtype = np.uint8) # Initialize numpy array
    for i in range(N):
        for j in range(rows*cols):
            tmp_pixel = images.read(1) # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]

    images.close()
    labels.close()
    return (x, y)
  
  X_train, y_train = loadmnist('~/data/course_data/t10k-images-idx3-ubyte','~/data/course_data/t10k-labels-idx1-ubyte')
  X_test, y_test = loadmnist('~/data/course_data/train-images-idx3-ubyte','~/data/course_data/train-labels-idx1-ubyte')
  
  X_train.shape
  """
  (10000, 784)
  """
  
  X_test
  """
  array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
  """
  
  y_test
  """
  array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
  """
  
  plt.figure(figsize = (20, 4))
for index, (image, label) in enumerate(zip(X_train[0:5], y_train[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (28, 28)))
    plt.title('Traininng: %i\n' % label, fontsize = 20)
```

![image-20210831231306773](http://qiniu.hivan.me/picGo/20210831231306.png?imgNote)

We only choose label with 0 and 6

```python
zero_train_indices = np.where(y_train == 0)
one_train_indices = np.where(y_train == 6)
train_indices = np.concatenate((zero_train_indices[0], one_train_indices[0]))

zero_test_indices = np.where(y_test == 0)
one_test_indices = np.where(y_test == 6)
test_indices = np.concatenate((zero_test_indices[0], one_test_indices[0]))

train_indices = np.random.choice(train_indices, size = len(train_indices), replace=False)
test_indices = np.random.choice(test_indices, size = len(test_indices), replace=False)

val_ratio= 0.2

train_indices = train_indices[: int(len(train_indices) * (1 - val_ratio))]
val_indices = train_indices[int(len(train_indices) * (1 - val_ratio)):]

binary_x_train = X_train[train_indices]
binary_x_test = X_test[test_indices]
binary_x_val = X_train[val_indices]

binary_y_train = y_train[train_indices]
binary_y_test = y_test[test_indices]
binary_y_val = y_train[val_indices]

import random
binary_y_train
"""
array([6, 0, 0, ..., 6, 0, 0], dtype=uint8)
"""

plt.imshow(np.reshape(binary_x_train[1], (28,28)))
plt.title('Training: %i\n' % binary_y_train[1], fontsize =20)
"""
Text(0.5, 1.0, 'Training: 0\n')
"""
```

![image-20210831231410866](http://qiniu.hivan.me/picGo/20210831231410.png?imgNote)



```python
from collections import Counter
Counter(binary_y_train)
"""
Counter({6: 768, 0: 782})
"""

Counter(binary_y_test)
"""
Counter({6: 5918, 0: 5923})
"""

Counter(binary_y_val)
"""
Counter({0: 148, 6: 162})
"""
```

#### Build model

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0, solver = 'lbfgs')
# L-BFGS-B - Software for Large-scale Bound-constrained Optimization

import warnings
warnings.filterwarnings('ignore')
clf.fit(binary_x_train, binary_y_train)
"""
LogisticRegression(random_state=0)
"""

clf.coef_
"""
array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
show more (open the raw output data in a text editor) ...

         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00]])
"""

clf.intercept_
"""
array([0.00016519])
"""

clf.score
"""
<bound method ClassifierMixin.score of LogisticRegression(random_state=0)>
"""

clf.score(binary_x_train, binary_y_train)
"""
1.0
"""

clf.score(binary_x_val, binary_y_val)
"""
1.0
"""

binary_x_test.shape
"""
(11841, 784)
"""

binary_y_test.shape
"""
0.9865720800608057
"""

predicated_result = clf.predict(binary_x_test)
np.where(binary_y_test != predicated_result)
"""
(array([   17,    45,    66,   137,   260,   279,   323,   453,   529,
          739,   753,   947,  1034,  1248,  1290,  1422,  1434,  1444,
 				...
        10677, 10739, 10750, 10979, 11010, 11058, 11104, 11113, 11366,
        11389, 11421, 11458, 11528, 11659, 11760]),)
"""

lookup_index = 1184

plt.imshow(np.reshape(binary_x_test[lookup_index], (28, 28)))
plt.title('Actual Value: {} ; Predict Value: {} \n'.format(binary_y_test[lookup_index], predicated_result[lookup_index]), fontsize = 20)
"""
Text(0.5, 1.0, 'Actual Value: 6 ; Predict Value: 6 \n')
"""
```

![image-20210831231740052](http://qiniu.hivan.me/picGo/20210831231740.png?imgNote)

```python
from sklearn import metrics
binary_y_test[0]
"""
6
"""

predicated_result[0]
"""
6
"""

metrics.precision_score(binary_y_test, predicated_result, average = 'macro')
"""
0.9865879016517065
"""

metrics.precision_score(binary_y_test, predicated_result, pos_label = 6)
"""
0.9837056946077608
"""

metrics.recall_score(binary_y_test, predicated_result, pos_label = 6)
"""
0.9895234876647516
"""

fpr, tpr, threshold = metrics.roc_curve(binary_y_test, predicated_result, pos_label = 6)
metrics.auc(fpr, tpr)
"""
0.9865733258009728
"""

cm = metrics.confusion_matrix(binary_y_test, predicated_result)

import seaborn as sns
from sklearn.metrics import confusion_matrix
data = confusion_matrix(binary_y_test, predicated_result)
data
"""
array([[5826,   97],
       [  62, 5856]])
"""

df_cm = pd.DataFrame(data, columns = np.unique(binary_y_test), index = np.unique(binary_y_test))

plt.figure(figsize = (10, 7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws = {'size': 16}) # font size
"""
<AxesSubplot:>
"""
```

![image-20210831232005063](http://qiniu.hivan.me/picGo/20210831232005.png?imgNote)

```python
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10, 10))
sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={'size': 16})
"""
<AxesSubplot:xlabel='Predicted', ylabel='Actual'>
"""
```

![image-20210831232124003](http://qiniu.hivan.me/picGo/20210831232124.png?imgNote)



## Boston code reproduction and reference answers

```python
# Import package
# Used to load the Boston housing price data set
from sklearn.datasets import load_boston
# pandas toolkit For students who are new to pandas, please refer to the official 10-minute tutorial: https://pandas.pydata.org/pandas-docs/stable/10min.html
import pandas as pd
# seaborn for drawing
import seaborn as sns
import numpy as np # numpy
# Show drawing
%matplotlib inline

data = load_boston()
data.keys()
"""
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
"""

df = pd.DataFrame(data['data'])
df.head()
"""
				0		1			2			3			4			5		6				7			8			9		10			11		12
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98
1	0.02731	 0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14
2	0.02729  0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03
3	0.03237  0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94
4	0.06905  0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33
"""

data['feature_names']
"""
array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
"""
```



### Field meaning

| **名称**    | 中文描述                                                 |
| ----------- | -------------------------------------------------------- |
| **CRIM**    | 住房所在城镇的人均犯罪率                                 |
| **ZN**      | 住房用地超过 25000 平方尺的比例                          |
| **INDUS**   | 住房所在城镇非零售商用土地的比例                         |
| **CHAS**    | 有关查理斯河的虚拟变量（如果住房位于河边则为1,否则为0 ） |
| **NOX**     | 一氧化氮浓度                                             |
| **RM**      | 每处住房的平均房间数                                     |
| **AGE**     | 建于 1940 年之前的业主自住房比例                         |
| **DIS**     | 住房距离波士顿五大中心区域的加权距离                     |
| **RAD**     | 离住房最近的公路入口编号                                 |
| **TAX**     | **每** **10000** **美元的全额财产税金额**                |
| **PTRATIO** | 住房所在城镇的师生比例                                   |
| **B**       | 1000(Bk-0.63)^2,其中 Bk 指代城镇中黑人的比例             |
| **LSTAT**   | 弱势群体人口所占比例                                     |
| **MEDV**    | 业主自住房的中位数房价（以千美元计）                     |

```python
df.columns = data['feature_names']
df.head()
"""
		CRIM		ZN	INDUS	CHAS		NOX		RM	AGE			DIS		RAD	TAX	PTRATIO		B		LSTAT
0	0.00632	18.0	2.31	0.0		0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98
1	0.02731	0.0		7.07	0.0		0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14
2	0.02729	0.0		7.07	0.0		0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03
3	0.03237	0.0		2.18	0.0		0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94
4	0.06905	0.0		2.18	0.0		0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33
"""

df['price'] = data['target']
df.head(2)
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	price
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.9	4.98	24.0
1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.9	9.14	21.6
"""

sns.heatmap(df.corr(), annot=True, fmt='.1f')
```

![image-20210831233019474](http://qiniu.hivan.me/picGo/20210831233019.png?imgNote)



```python
import matplotlib.pyplot as plt
plt.scatter(df['RM'], df['price'])
"""
<matplotlib.collections.PathCollection at 0x7fe0f984f810>
"""
```

![image-20210831233046720](http://qiniu.hivan.me/picGo/20210831233046.png?imgNote)

```python
plt.figure(figsize = (20, 5))

features = ['LSTAT', 'RM']
target = df['price']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker = 'o')
    plt.title('{} vs price'.format(col))
    plt.xlabel(col)
    plt.ylabel('price')
```

![image-20210831233103355](http://qiniu.hivan.me/picGo/20210831233103.png?imgNote)

```python
x = df['RM']
y = df['price']

history_notes = {_x: _y for _x, _y in zip(x, y)}
history_notes[6.575]
"""
24.0
"""

# Find the top three prices closest to RM:6.57,
similary_ys = [y for _, y in sorted(history_notes.items(), key=lambda x_y: (x_y[0]-6.57) ** 2)[:3]]
similary_ys
"""
[23.8, 24.0, 24.8]
"""

np.mean(similary_ys) # Calculate the average of three
"""
24.2
"""
```

Using historical data to predict data that has never been seen before, the most direct method

### K-Neighbor-Nearst

```python
def knn(query_x, history, top_n=3):
    sorted_notes = sorted(history.items(), key=lambda x_y: (x_y[0] - query_x) ** 2) 
    similar_notes = sorted_notes[:top_n]
    similar_ys = [y for _, y in similar_notes]
    
    return np.mean(similar_ys)

knn(5.4, history_notes)
"""
15.700000000000001
"""
```

In order to obtain results faster, we hope to obtain predictive power by fitting a function
$$
f(rm) = k * rm + b
$$


Random Approach

$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} (\hat{y_i} - y_i) ^ 2 $$

$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} ((k * rm_i + b) - y_i) ^ 2 $$

```python
def loss(yhat, y):
    return np.mean((yhat - y) **2)
import random

min_loss = float('inf')
best_k, bes_b = None, None

print(min_loss)

min_loss = float('inf')
best_k, bes_b = None, None

for step in range(1000):
    min_v, max_v = -100, 100
    k, b = random.randrange(min_v, max_v), random.randrange(min_v, max_v)
    y_hats = [k * rm_i + b for rm_i in x]
    current_loss = loss(y_hats, y)
    
    if current_loss <min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('In step {}, we have obtained the function f(rm) = {} * rm + {}, at this time loss is: {}'.format(step, k, b, current_loss))
"""
In step 0, we have obtained the function f(rm) = 14 * rm + -78, at this time loss is: 212.87040239525695
In step 70, we have obtained the function f(rm) = 10 * rm + -47, at this time loss is: 88.70654683794466
In step 256, we have obtained the function f(rm) = 13 * rm + -55, at this time loss is: 68.45390542094862
In step 526, we have obtained the function f(rm) = 10 * rm + -37, at this time loss is: 54.977297826086954
"""

plt.scatter(x, y)
plt.scatter(x, [best_k * rm + best_b for rm in x])
"""
<matplotlib.collections.PathCollection at 0x7fe0980f37d0>
"""
```

![image-20210831233425089](http://qiniu.hivan.me/picGo/20210831233425.png?imgNote)



### Monte Carlo simulation

#### Supervisor

$$ Loss(k, b) = \frac{1}{n} \sum_{i \in N} ((k * rm_i + b) - y_i) ^ 2 $$

$$ \frac{\partial{loss(k, b)}}{\partial{k}} = \frac{2}{n}\sum_{i \in N}(k * rm_i + b - y_i) * rm_i $$

$$ \frac{\partial{loss(k, b)}}{\partial{b}} = \frac{2}{n}\sum_{i \in N}(k * rm_i + b - y_i)$$



```python
def partial_k(k, b, x, y):
    return 2 * np.mean((k*x+b-y) *x)
def partial_b(k, b, x, y):
    return 2*np.mean(k*x+b-y)
  
k, b = random.random(), random.random()
min_loss = float('inf')
best_k, best_b = None, None
learning_rate = 1e-2

for step in range(2000):
    k,b = k+(-1*partial_k(k,b,x,y) * learning_rate), b+(-1*partial_b(k,b,x,y) * learning_rate)
    y_hats = k * x +b
    current_loss = loss(y_hats, y)

    if current_loss < min_loss:
        min_loss = current_loss
        best_k, best_b = k, b
        print('On the {} step, we have func f(rm) = {} * rm + {}, loss is {} now'.format(step, k, b, current_loss))
"""
On the 0 step, we have func f(rm) = 6.968714597804018 * rm + -21.099847342593957, loss is 45.86961514375004 now
On the 1 step, we have func f(rm) = 6.9692276199804555 * rm + -21.103110737199852, loss is 45.86852398135223 now
show more (open the raw output data in a text editor) ...

On the 1999 step, we have func f(rm) = 7.783005326604901 * rm + -26.279646762684518, loss is 44.468037178267025 now
"""

best_k, best_b
"""
(10, -37)
"""

plt.scatter(x, y)
plt.scatter(x, [best_k * rm + best_b for rm in x])
```

![image-20210831233620249](http://qiniu.hivan.me/picGo/20210831233620.png?imgNote)



#### Supervised Learning

We turn the forecast of housing prices into a more responsible and sophisticated model. What should we do?

$$ f(x) = k * x + b $$

$$ f(x) = k2 * \sigma(k_1 * x + b_1) + b2 $$

$$ \sigma(x) = \frac{1}{1 + e^(-x)} $$ 



```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sub_x = np.linspace(-10, 10)
plt.plot(sub_x, sigmoid(sub_x))
```

![image-20210831233717648](http://qiniu.hivan.me/picGo/20210831233717.png?imgNote)



```python
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

![image-20210831233740329](http://qiniu.hivan.me/picGo/20210831233740.png?imgNote)

We can implement more complex functions through simple, basic modules and repeated superposition

For more and more complex functions? How does the computer seek guidance?

1. What is machine learning?
2. The shortcomings of the KNN method, what is the background of the proposed linear fitting
3. How to obtain faster function weight update through supervision method
4. The combination of nonlinear and linear functions can fit very complex functions
5. Deep learning we can fit more complex functions through basic function modules

#### Assigment

$$ L2-Loss(y, \hat{y}) = \frac{1}{n}\sum{(\hat{y} - y)}^2 $$

$$ L1-Loss(y, \hat{y}) = \frac{1}{n}\sum{|(\hat{y} - y)|} $$

Change L2-Loss in the code to L1Loss and implement gradient descent

Realize L1Loss gradient descent from 0

##### 1 Import package

```python
import numpy as np
import pandas as pd
```

##### 2 Load data set

```python
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
"""
dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
"""

X = boston.data
y = boston.target

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df.head()
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
0	0.00632	18.0	2.31	0.0	0.538	6.575	65.2	4.0900	1.0	296.0	15.3	396.90	4.98
1	0.02731	0.0	7.07	0.0	0.469	6.421	78.9	4.9671	2.0	242.0	17.8	396.90	9.14
2	0.02729	0.0	7.07	0.0	0.469	7.185	61.1	4.9671	2.0	242.0	17.8	392.83	4.03
3	0.03237	0.0	2.18	0.0	0.458	6.998	45.8	6.0622	3.0	222.0	18.7	394.63	2.94
4	0.06905	0.0	2.18	0.0	0.458	7.147	54.2	6.0622	3.0	222.0	18.7	396.90	5.33
"""

df.describe() # Data description, you can view the statistics of each variable
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
count	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000
mean	3.613524	11.363636	11.136779	0.069170	0.554695	6.284634	68.574901	3.795043	9.549407	408.237154	18.455534	356.674032	12.653063
std	8.601545	23.322453	6.860353	0.253994	0.115878	0.702617	28.148861	2.105710	8.707259	168.537116	2.164946	91.294864	7.141062
min	0.006320	0.000000	0.460000	0.000000	0.385000	3.561000	2.900000	1.129600	1.000000	187.000000	12.600000	0.320000	1.730000
25%	0.082045	0.000000	5.190000	0.000000	0.449000	5.885500	45.025000	2.100175	4.000000	279.000000	17.400000	375.377500	6.950000
50%	0.256510	0.000000	9.690000	0.000000	0.538000	6.208500	77.500000	3.207450	5.000000	330.000000	19.050000	391.440000	11.360000
75%	3.677083	12.500000	18.100000	0.000000	0.624000	6.623500	94.075000	5.188425	24.000000	666.000000	20.200000	396.225000	16.955000
max	88.976200	100.000000	27.740000	1.000000	0.871000	8.780000	100.000000	12.126500	24.000000	711.000000	22.000000	396.900000	37.970000
"""
```

##### 3 Data preprocessing

Normalization or standardization can prevent a certain dimension or a few dimensions from affecting the data too much when there are very many dimensions, and secondly, the program can run faster. There are many methods, such as standardization, min-max, z-score, p-norm, etc. How to use it depends on the characteristics of the data set.

[Extended reading-the deep learning field of the myth of data standardization](https://zhuanlan.zhihu.com/p/81560511)

```python



from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # z = (x-u) / s u is the mean, s is the standard deviation
X = ss.fit_transform(df) # For linear models, normalization or standardization is generally required, otherwise there will be a gradient explosion, which is generally not required for tree models
df = pd.DataFrame(X, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX' ,'PTRATIO','B','LSTAT'])
df.describe()
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT
count	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02	5.060000e+02
mean	2.808469e-17	6.599903e-16	-4.633974e-16	-4.353127e-16	1.404235e-16	-1.755293e-17	2.176564e-16	-1.685082e-16	-5.055245e-16	8.987102e-16	-1.067218e-15	4.493551e-16	-2.246775e-16
std	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00	1.000990e+00
min	-4.197819e-01	-4.877224e-01	-1.557842e+00	-2.725986e-01	-1.465882e+00	-3.880249e+00	-2.335437e+00	-1.267069e+00	-9.828429e-01	-1.313990e+00	-2.707379e+00	-3.907193e+00	-1.531127e+00
25%	-4.109696e-01	-4.877224e-01	-8.676906e-01	-2.725986e-01	-9.130288e-01	-5.686303e-01	-8.374480e-01	-8.056878e-01	-6.379618e-01	-7.675760e-01	-4.880391e-01	2.050715e-01	-7.994200e-01
50%	-3.906665e-01	-4.877224e-01	-2.110985e-01	-2.725986e-01	-1.442174e-01	-1.084655e-01	3.173816e-01	-2.793234e-01	-5.230014e-01	-4.646726e-01	2.748590e-01	3.811865e-01	-1.812536e-01
75%	7.396560e-03	4.877224e-02	1.015999e+00	-2.725986e-01	5.986790e-01	4.827678e-01	9.067981e-01	6.623709e-01	1.661245e+00	1.530926e+00	8.065758e-01	4.336510e-01	6.030188e-01
max	9.933931e+00	3.804234e+00	2.422565e+00	3.668398e+00	2.732346e+00	3.555044e+00	1.117494e+00	3.960518e+00	1.661245e+00	1.798194e+00	1.638828e+00	4.410519e-01	3.548771e+00
"""
```

$$ y=Σwixi+b $$

Because the derivation of b is all 1, add a bias b to the data and set it to 1, as a feature of the data and update the gradient wi*b=wi

```python
df['bias'] = 1
df
"""
	CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	bias
0	-0.419782	0.284830	-1.287909	-0.272599	-0.144217	0.413672	-0.120013	0.140214	-0.982843	-0.666608	-1.459000	0.441052	-1.075562	1
1	-0.417339	-0.487722	-0.593381	-0.272599	-0.740262	0.194274	0.367166	0.557160	-0.867883	-0.987329	-0.303094	0.441052	-0.492439	1
2	-0.417342	-0.487722	-0.593381	-0.272599	-0.740262	1.282714	-0.265812	0.557160	-0.867883	-0.987329	-0.303094	0.396427	-1.208727	1
3	-0.416750	-0.487722	-1.306878	-0.272599	-0.835284	1.016303	-0.809889	1.077737	-0.752922	-1.106115	0.113032	0.416163	-1.361517	1
4	-0.412482	-0.487722	-1.306878	-0.272599	-0.835284	1.228577	-0.511180	1.077737	-0.752922	-1.106115	0.113032	0.441052	-1.026501	1
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
501	-0.413229	-0.487722	0.115738	-0.272599	0.158124	0.439316	0.018673	-0.625796	-0.982843	-0.803212	1.176466	0.387217	-0.418147	1
502	-0.415249	-0.487722	0.115738	-0.272599	0.158124	-0.234548	0.288933	-0.716639	-0.982843	-0.803212	1.176466	0.441052	-0.500850	1
503	-0.413447	-0.487722	0.115738	-0.272599	0.158124	0.984960	0.797449	-0.773684	-0.982843	-0.803212	1.176466	0.441052	-0.983048	1
504	-0.407764	-0.487722	0.115738	-0.272599	0.158124	0.725672	0.736996	-0.668437	-0.982843	-0.803212	1.176466	0.403225	-0.865302	1
505	-0.415000	-0.487722	0.115738	-0.272599	0.158124	-0.362767	0.434732	-0.613246	-0.982843	-0.803212	1.176466	0.441052	-0.669058	1
506 rows × 14 columns
"""
```

Divide the data set, where 20% of the data is used as the test set X_test, y_test, and the other 80% are used as the training set X_train, y_train, where random_state is the random seed

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(df, y, test_size = 0.2, random_state = 42)

print('X_train.shape, y_train.shape:', X_train.shape, y_train.shape)
print('X_test.shape, y_test.shape', X_test.shape, y_test.shape)
"""
X_train.shape, y_train.shape: (404, 14) (404,)
X_test.shape, y_test.shape (102, 14) (102,)
"""

X_train = np.array(X_train)
```

#### Model training and gradient update

```python
def l1_cost(X, y, theta):
    """
    X: 特征
    y: 目标值
    theta: 模型参数
    """
    k = X.shape[0]
    total_cost = 0
    for i in range(k):
        total_cost =+ 1/k * np.abs(y[i] - theta.dot(X[i, :]))
    return total_cost
  
def l2_cost(X, y, theta):
    k = X.shape[0]
    total_cost = 0
    for i in range(k):
        total_cost += 1/k * (y[i] - theta.dot(X[i, :])) ** 2
    return total_cost
  
np.zeros(10).shape
"""
(10,)
"""

def step_l1_gradient(X, y, learning_rate, theta):
    """
    Function to calculate the gradient of the MAE loss function
    Return the gradient value 0 for the non-differentiable point at 0
    X: feature vector
    y: target value
    learing_rate: learning rate
    theta: parameter
    """

    n = X.shape[0]
    print(n)
    e = y-X @ theta
    gradients = -(X.T @ np.sign(e)) / n
    theta = theta-learning_rate * gradients
    return theta
  
def step_l2_gradient(X, y, learning_rate, theta):
    k = X.shape[0]
    x = X.shape[1]
    gradients = np.zeros(n)
    for i in range(k):
        for j in range(n):
            gradients[j] += (-2/k) * (y[i] - (theta.dot(X[i, :]))) * X[i, j]
    theta = theta - learning_rate * gradients
    return theta
  
def step_gradient(X, y, learning_rate, theta):
    """
    X: feature vector
    y: target value
    learing_rate: learning rate
    theta: parameter
    """
    m_deriv = 0
    N = len(X)
    for i in range(N):
        # Calculate the partial derivative
        # -x(y-(mx + b)) / |mx + b|
        m_deriv +=-X[i] * (y[i]-(theta*X[i] + b)) / abs(y[i]-(theta*X[i] + b))
    # We subtract because the derivatives point in direction of steepest ascent
    theta -= (m_deriv / float(N)) * learning_rate
# theta = theta-learning_rate * gradients
    return theta

def gradient_descent(X_train, y_train, learning_rate, iterations):
    k = X_train.shape[0]
    n = X_train.shape[1]
    theta = np.zeros(n)
    loss_values = []
    print(theta.shape)
    for i in range(iterations):
        theta = step_l1_gradient(X_train, y_train, learning_rate, theta)
        loss = l1_cost(X_train, y_train, theta)
        loss_values.append(loss)
        print(i, 'cost:', loss)
    return theta, loss_values
  
# Training parameters
learning_rate = 0.04 # Learning rate
iterations = 300 # number of iterations
theta ,loss_values = gradient_descent(X_train, y_train, learning_rate, iterations)
"""
(14,)
404
0 cost: 0.04594399172713912
404
1 cost: 0.045848379493882215
404
show more (open the raw output data in a text editor) ...

299 cost: 0.017838215258874083
"""
```

## Heart Practise

```python
import pandas as pd
path = '~/data/'
dataPath = path + 'heart.csv'
train_data = pd.read_csv(dataPath)
```

### Field meaning

| **字段名**   | 含义                                                         |
| ------------ | ------------------------------------------------------------ |
| **age**      | 年龄                                                         |
| **sex**      | 性别(1 = 男性, 0 = 女性)                                     |
| **cp**       | 胸部疼痛类型(值1：典型心绞痛，值2：非典型性心绞痛，值3：非心绞痛，值4：无症状） |
| **trestbps** | 血压                                                         |
| **chol**     | 胆固醇                                                       |
| **fbs**      | 空腹血糖（> 120 mg/dl，1=真；0=假）                          |
| **restecg**  | 心电图结果（0=正常，1=患有ST-T波异常，2=根据Estes的标准显示可能或确定的左心室肥大） |
| **thalach**  | 最大心跳数                                                   |
| **exang**    | 运动时是否心绞痛（1=有过；0=没有）                           |
| **oldpeak**  | **运动相对于休息的****ST**                                   |
| **slop**     | 心电图ST segment的倾斜度(值1:上坡，值2:平坦，值3:下坡）      |
| **ca**       | 透视检查看到的血管数                                         |
| **thal**     | 缺陷种类（3=正常；6=固定缺陷；7=可逆缺陷）                   |
| **target**   | 是否患病（0=否，1=是）                                       |

### Print a brief summary of the data set

```python
train_data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
"""

train_data.target.value_counts()
"""
1    165
0    138
Name: target, dtype: int64
"""
```

Change the "sex" column to two columns "sex_0" and "sex_1".

```python
sex = pd.get_dummies(train_data['sex'], prefix = "sex") 
```

Add "sex_0" and "sex_1" to the data set.

```python
train_data = pd.concat([train_data,sex], axis = 1) 
```

And delete the sex column

```python
train_data = train_data.drop(columns = ['sex'])
```

Print out the first five lines. Check whether sex_0, sex_1 are added successfully, and whether sex is deleted successfully.

```python
train_data.head()
"""
	age	cp	trestbps	chol	fbs	restecg	thalach	exang	oldpeak	slope	ca	thal	target	sex_0	sex_1
0	63	3			145			233			1			0				150			0			2.3		0		0			1				1			0			1
1	37	2			130			250			0			1				187			0			3.5		0		0			2				1			0			1
2	41	1			130			204			0			0				172			0			1.4		2		0			2				1			1			0
3	56	1			120			236			0			1				178			0			0.8		2		0			2				1			0			1
4	57	0			120			354			0			1				163			1			0.6		2		0			2				1			1			0
"""
```

Get sample label

```pyhon
y_data = train_data.target.values
train_data.shape
"""
(303, 15)
"""
```

Get sample feature set

```python
x_data = train_data.drop(['target'],axis=1)
x_data.shape
"""
(303, 14)
"""
```

Divide the data set Parameters: `test_size=0.3, random_state=33`

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3, random_state=33)
```

### Normalization

 Import the StandardScaler package and initialize

```python
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
```

fit function/module is used to train model parameters

```python
standardScaler.fit(X_train)
```

Standardize the training set and test set

```python
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test) 
```

Define logistic regression model

```python
from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
```

Calculate the training set score

```python
log_reg.score(X_train,y_train)
"""
0.8537735849056604
"""
```

Calculate the test set score

```python
log_reg.score(X_test,y_test)
"""
0.8461538461538461
"""
```

Use the classification_report function to display a text report of the main classification indicators

```python
from sklearn.metrics import classification_report
y_predict_log = log_reg.predict(X_test)
print(classification_report(y_test,y_predict_log))
"""
 precision    recall  f1-score   support

           0       0.93      0.78      0.85        50
           1       0.78      0.93      0.84        41

    accuracy                           0.85        91
   macro avg       0.85      0.85      0.85        91
weighted avg       0.86      0.85      0.85        91
"""
```

