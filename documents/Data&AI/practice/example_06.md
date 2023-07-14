# Machine Learning Part-04

## SVM
```python
import numpy as np

label_a = np.random.normal(6, 2, size = (50,2))
label_b = np.random.normal(-6, 2, size = (50,2))

import matplotlib.pyplot as plt

a = [1, 2, 3]
b = [-1,-2, -3]

plt.scatter(*zip(*label_a))
plt.scatter(*zip(*label_b))
plt.show()
```

![image-20210901170815036](http://qiniu.hivan.me/MAMTimage-20210901170815036.png?img)



```python
label_a_x = label_a[:, 0]
label_b_x = label_b[:, 0]

def f(x, k, b):
    return k*x -b
  
k_and_b = []
for i in range(100):
    k, b = (np.random.random(size = (1,2)) * 10 - 5)[0]

    if np.max(f(label_a_x, k, b)) <= -1 and np.min(f(label_b_x, k, b)) >= 1:
        print(k, b)
        k_and_b.append((k, b))
"""
-3.4732670434285517 -2.3248316389039325
-3.654276254462583 0.01110189858052646
-2.4609031871010014 -0.3932180655739925
-2.9206497777762843 0.2595456609552631
-4.07589152330003 -0.6463313059119606
-3.1950366475236835 -1.8558958669742989
-4.316670785852706 -3.1033030808371653
-4.124339773909792 -1.5741734685470687
-4.20817621470405 0.4368323022696625
-3.7098120657624003 -0.38196175566618784
-3.2053446683533315 0.12822700803583054
-4.534694169094692 1.143734501297419
-4.8124714209376425 0.8707258703100704
"""

plt.scatter(*zip(*label_a))
plt.scatter(*zip(*label_b))

for k, b in k_and_b:
    x = np.concatenate((label_a_x, label_b_x))
    plt.plot(x, f(x, k, b))

plt.show()
```

![image-20210901170902373](http://qiniu.hivan.me/MAMTimage-20210901170902373.png?img)



```python
plt.scatter(*zip(*label_a))
plt.scatter(*zip(*label_b))

k,b = sorted(k_and_b, key = lambda t: abs(t[0]))[0]
x = np.concatenate((label_a_x, label_b_x))
plt.plot(x, f(x, k, b))
plt.show()
```

![image-20210901170921575](http://qiniu.hivan.me/MAMTimage-20210901170921575.png?img)



```python
from sklearn.datasets import load_boston

datasets = load_boston()
data, target = datasets['data'], datasets['target']

import pandas as pd

df = pd.DataFrame(data)
df.columns = datasets['feature_names']

import random

def random_select(df, drop_num = 4):
    columns = random.sample(list(df.columns), k = len(df.columns) - drop_num)

    return df[columns]
  
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

sample_x = random_select(df)
regressioner = DecisionTreeRegressor()
(X_train, X_test, y_train, y_test)  = train_test_split(sample_x, target, test_size = 0.3)
regressioner.fit(X_train, y_train)
regressioner.score(X_train, y_train)
"""
1.0
"""

regressioner.score(X_test, y_test)
"""
0.8110635350395325
"""

def random_tree(train_X, train_y, test_X, test_y, drop_n = 4):
    train_sample = random_select(train_X, drop_num = drop_n)

    regressioner = DecisionTreeRegressor()
    regressioner.fit(train_sample, train_y)

    train_score = regressioner.score(train_sample, train_y)
    test_score = regressioner.score(test_X[train_sample.columns], test_y)

    print('train score = {}; test score = {}'.format(train_score, test_score))

    y_predicat = regressioner.predict(test_X[train_sample.columns])

    return y_predicat
  
def random_forest(train_X, train_y, test_X, test_y, tree_n = 4):
    predicat = np.array([random_tree(train_X, train_y, test_X, test_y) for _ in range(tree_n)])

    return np.mean(predicat, axis = 0)
  
(X_train, X_test, y_train, y_test)  = train_test_split(df, target, test_size = 0.3)

forest_predict = random_forest(X_train, y_train, X_test, y_test)
"""
train score = 1.0; test score = 0.5367061884031776
train score = 1.0; test score = 0.4983695562874999
train score = 1.0; test score = 0.6715869370883646
train score = 1.0; test score = 0.6210922529610217
"""

forest_predict
"""
array([10.925, 21.1  , 30.625, 28.025, 22.525, 17.65 , 20.6  , 17.325,
       29.175, 14.95 , 40.775, 19.55 , 12.175, 23.675, 10.775, 22.1  ,
       ...
       15.575, 20.5  , 22.775, 30.725, 18.975, 16.45 , 22.05 , 18.925])
"""

from sklearn.metrics import r2_score
r2_score(y_test, forest_predict)
"""
0.7840500839091215
"""
```



##  Entropy: 熵

```python
import numpy as np
from collections import Counter
from icecream import ic
from functools import lru_cache

def pr(es):
    counter = Counter(es)
    def _wrap(e):
        return counter[e] / len(es)
    return _wrap

def entropy(elements):
    # Information Entropy
    p = pr(elements)
    return -np.sum(p(e) * np.log(p(e)) for e in set(elements))


def gini(elements):
    p = pr(elements)
    return 1-np.sum(p(e) ** 2 for e in set(elements))
  
pure_func = gini

ic(pure_func([1, 1, 1, 1, 1, 0]))
ic(pure_func([1, 1, 1, 1, 1, 1]))
ic(pure_func([1, 2, 3, 4, 5, 8]))
ic(pure_func([1, 2, 3, 4, 5, 9]))
ic(pure_func(['a', 'b', 'c', 'c', 'c', 'c', 'c']))
ic(pure_func(['a', 'b', 'c', 'c', 'c', 'c', 'd']))
"""
ic| pure_func([1, 1, 1, 1, 1, 0]): 0.2777777777777777
ic| pure_func([1, 1, 1, 1, 1, 1]): 0.0
ic| pure_func([1, 2, 3, 4, 5, 8]): 0.8333333333333333
ic| pure_func([1, 2, 3, 4, 5, 9]): 0.8333333333333333
ic| pure_func(['a', 'b', 'c', 'c', 'c', 'c', 'c']): 0.44897959183673464
ic| pure_func(['a', 'b', 'c', 'c', 'c', 'c', 'd']): 0.6122448979591837
0.6122448979591837
"""
```



## Random forest

```python
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


house = load_boston()
X = house.data
y = house.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_train, y_train)

print('whole dataset train acc: {}'.format(tree_reg.score(x_train, y_train)))
print('whole dataset test acc: {}'.format(tree_reg.score(x_test, y_test)))
"""
whole dataset train acc: 1.0
whole dataset test acc: 0.6776520888466615
"""

def random_forest(train_x, train_y, test_x, test_y, drop_n=4):
    random_features = np.random.choice(list(train_x.columns), size=len(train_x.columns)-drop_n)

    sample_x = train_x[random_features]
    sample_y = train_y

    reg = DecisionTreeRegressor()
    reg.fit(sample_x, sample_y)

    train_score = reg.score(sample_x, sample_y)
    test_score = reg.score(test_x[random_features], test_y)

    print('sub sample :: train score: {}, test score: {}'.format(train_score, test_score))

    y_predicated = reg.predict(test_x[random_features])

    return y_predicated, test_score
  
with_feature_names = pd.DataFrame(X)
with_feature_names.columns = house['feature_names']

x_train, x_test, y_train, y_test = train_test_split(with_feature_names, y, test_size=0.3, random_state=0)

tree_num = 4
predicates = []
for _ in range(tree_num):
    predicated, score = random_forest(x_train, y_train, x_test, y_test)
    predicates.append((predicated, score))
"""
sub sample :: train score: 1.0, test score: 0.5640870175410873
sub sample :: train score: 1.0, test score: 0.29024437819534354
sub sample :: train score: 1.0, test score: 0.37812117132843814
sub sample :: train score: 1.0, test score: 0.5650888856735524
"""

predicates_value = [v for v, s in predicates]
forest_scores = [s for v, s in predicates]

print('the score of forest is : {}'.format(r2_score(y_test, np.mean(predicates_value, axis=0))))
"""
the score of forest is : 0.680193104551715
"""

weights = np.array(forest_scores) / np.sum(forest_scores)

weights_score = np.zeros_like(np.mean(predicates_value, axis=0))
for i, v in enumerate(predicates_value):
    weights_score += v * weights[i]
    
print('the score of weighted forest is : {}'.format(r2_score(y_test, weights_score)))
"""
the score of weighted forest is : 0.6956613076019385
"""


```



## Show SVM

```python
import numpy as np
import matplotlib.pyplot as plt

label_a = np.random.normal(6, 2, size=(50, 2))
label_b = np.random.normal(-6, 2, size=(50, 2))

plt.scatter(*zip(*label_a))
plt.scatter(*zip(*label_b))

label_a_x = label_a[:, 0]
label_b_x = label_b[:, 0]

def f(x, w, b):
    return w * x + b
  
k_and_b = []
for i in range(100):
    k, b = (np.random.random(size=(1, 2)) * 10 - 5)[0]

    if np.max(f(label_a_x, k, b)) >= -1 and np.min(f(label_b_x, k, b)) >= 1:
        print(k, b)
        k_and_b.append((k, b))
"""
0.17732109082579406 3.9508645615428843
-0.8649868307954458 1.7349996177756957
...
-2.2969567032985783 2.171321001904926
"""

for k, b in k_and_b:
    x = np.concatenate((label_a_x, label_b_x))
    plt.plot(x, f(x, k, b))
    
print(k_and_b)
"""
[(0.17732109082579406, 3.9508645615428843), (-0.8649868307954458, 1.7349996177756957), (-0.818317924604357, 0.352843348193578), (-0.19730603224472976, 4.002168852007262), 
...
(-2.2969567032985783, 2.171321001904926)]
"""

w, b = min(k_and_b, key=lambda k_b: k_b[0])

all_x = np.concatenate((label_a_x, label_b_x))
plt.plot(all_x, f(all_x, w, b), 'r-o')

plt.show()
```

![image-20210901171726065](http://qiniu.hivan.me/MAMTimage-20210901171726065.png?img)



## Integrated learning

Ensemble learning is a machine learning paradigm that solves the same problem by training multiple models. In contrast to ordinary machine learning methods that try to learn a hypothesis from training data, ensemble methods try to construct a set of hypotheses and use them in combination. Next, we will use the decision tree and its integrated version to model the classic data set Mnist and observe the differences in different integration methods.

```python
!ls
!unzip mnist_test.csv.zip && unzip mnist_train.csv.zip

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
```

#### Build a data set

The Mnist data set used this time is not in the original format. In order to more easily adapt to this training, the 28 * 28 pictures in the original data set are [flatten](https://numpy.org/doc/stable/reference /generated/numpy.ndarray.flatten.html) operation, it becomes 784 features, the columns in the DataFrame below: 1x1, 1x2, ..., 28x28, representing the *i* row and *j* column in the picture The pixel value of is a grayscale image, so the pixel value is only 0 and 1

```python
train_df = df = pd.read_csv('~/data/mnist_train.csv')
train_df.head()
"""
	label	1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
0	5	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	9	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 785 columns
"""
```

View training data information:, whether there is NaN, how many pieces of data are there...

```python
train_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60000 entries, 0 to 59999
Columns: 785 entries, label to 28x28
dtypes: int64(785)
memory usage: 359.3 MB
"""

test_df = df = pd.read_csv('~/data/mnist_test.csv')
test_df.head()
"""
	label	1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
0	7	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	2	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 785 columns
"""

test_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Columns: 785 entries, label to 28x28
dtypes: int64(785)
memory usage: 59.9 MB
"""
```

Build training and test data

```python
X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]

X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)
"""
(((60000, 784), (60000,)), ((10000, 784), (10000,)))
"""
```

#### Decision Tree

First train a simple decision tree to see how it performs

```python
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

dtc.score(X_train, y_train)
"""
1.0
"""

dtc.score(X_test, y_test)
"""
0.8753
"""

dtc = DecisionTreeClassifier(min_samples_leaf=8)
dtc.fit(X_train, y_train)

dtc.score(X_train, y_train), dtc.score(X_test, y_test)
"""
(0.9311666666666667, 0.8795)
"""
```

From the above results, we can see that by adjusting the parameter `min_samples_leaf`, the overfitting situation has been alleviated. What does this parameter mean? Why increasing it can alleviate the overfitting problem? The meaning of `min_samples_leaf` is the minimum number of samples contained in the leaf nodes of the decision tree. By increasing this parameter, the decision tree can not capture any of the subtle features of the training data during training, resulting in excessive training data. Fitting: The large number of samples of leaf nodes can also play a role in voting and enhance the generalization performance of the model. You can try to continue to increase the value of this parameter and try to find the best parameter. In addition to this parameter, you can also try to adjust the parameters such as `min_samples_split` and `max_features`. For the specific meaning, please refer to [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html )



### **Second question: **

**Try to adjust other parameters to see the performance of the decision tree on the test set**

#### Random Forest

Take a look at the bagging version of the decision tree and how the random forest performs!

```python
rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(X_train, y_train)

rfc.score(X_train, y_train), rfc.score(X_test, y_test)
"""
(0.99905, 0.9513)
"""
```

It is worthy of the integrated version. It basically achieves better performance under the default parameters. The accuracy of the test set is about 7% higher than that of the ordinary decision tree. However, comparing the training and test results, it can be found that there is still a certain degree of overfitting. , Try to adjust some parameters below

```python
rfc = RandomForestClassifier(n_estimators = 20)
rfc.fit(X_train, y_train)

rfc.score(X_train, y_train), rfc.score(X_test, y_test)
"""
(0.9999, 0.96)
"""
```

After increasing the parameter `n_estimators`, the accuracy of the test set has increased by about 1%. The meaning of this parameter is to train 20 decision trees at the same time, and finally integrate the results. The increase of this parameter can be simply regarded as voting The number of people increases, so the final result will inevitably be more robust. You can try to continue to increase this parameter, or adjust other parameters such as `max_samples`, appropriately less than the total amount of training data, which can increase the difference between different sub-models and further improve the generalization performance. It can also adjust the parameters of the base learner (decision tree). For the meaning of the parameters, see [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

#### GBDT

Let's compare the performance of the boosting version of the decision tree GBDT!

```python
gbc = GradientBoostingClassifier(n_estimators=10)
gbc.fit(X_train, y_train)

gbc.score(X_train, y_train), gbc.score(X_test, y_test)
"""
(0.8423, 0.846)
"""
```

As expected, the performance has been greatly improved, and the indicators of the training set are basically the same as those of the test set, and there is no overfitting, so it should be possible to continue to try to improve this parameter. Generally, in the absence of overfitting, we only need to consider continuing to increase the complexity of the model. This is the fastest way to improve performance. When the complexity of the model increases to the point of over-fitting, we then consider using some methods to reduce over-fitting.



#### Bagging

The aforementioned random forest and GBDT are ensemble learning algorithms based on decision trees, but it should be noted that ensemble learning is not exclusive to decision trees. Any other learner can be used as a base learner for ensemble learning, such as Logistic regression, support vector machine.


Bagging is short for "bootstrap aggregating". This is a meta-algorithm, which takes M sub-samples (with replacement) from the initial data set, and trains the prediction model on these sub-samples. The final model is obtained by averaging all sub-models, which usually produces better results. The main advantage of this technique is that it combines regularization, all you need to do is choose good parameters for the base learner.

The following uses the general api provided by sklearn to construct an integrated learning algorithm

```python
# Still use decision tree as base learner
bgc = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
bgc.fit(X_train, y_train)

bgc.score(X_train, y_train), bgc.score(X_test, y_test)
"""
(0.9935166666666667, 0.9506)
"""
```

### Third question

**Logistic regression as a base learner**

```python
bgc = BaggingClassifier(LogisticRegression(max_iter = 500), max_samples=0.5, max_features=1.0, n_estimators=20)
bgc.fit(X_train, y_train)

bgc.score(X_train, y_train), bgc.score(X_test, y_test)
"""
(0.9421166666666667, 0.9228)
"""
```

Above we have successfully used logistic regression as the base learner to complete integrated learning. You can try to use only logistic regression for training, and compare the performance of the single model with the bagging version of logistic regression.

#### Boosting

Boosting refers to a series of algorithms that can transform a weak learner into a strong learner. The main principle of boosting is to combine a series of weak learners (only better than random guessing). For those samples that were misclassified in the early stages of training, the boosting algorithm will give more attention. Then combine the predictions by weighted majority voting (classification) or weighted sum (regression) to produce the final prediction.

```python
abc = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=0.01)
abc.fit(X_train, y_train)
abc.score(X_train, y_train), abc.score(X_test, y_test)
"""
(1.0, 0.875)
"""
```

Comparing the boosting integrated version of decision tree and logistic regression, we can find that logistic regression has better generalization ability, and decision tree is easier to overfit

```python
abc = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=8), n_estimators=10, learning_rate=0.01)
abc.fit(X_train, y_train)
abc.score(X_train, y_train), abc.score(X_test, y_test)
"""
(0.9981833333333333, 0.9532)
"""
```

In fact, over-fitting is not a bad thing. If your model cannot be over-fitted, it means that it cannot fit the training data well. Therefore, the decision tree is very over-fitted at the beginning, which also shows its potential. , You can see that after the above parameters are adjusted, the boosting version of the decision tree easily exceeds the boosting version of the logistic regression













