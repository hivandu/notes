# Machine Learning Part-03

## Decision trees

Machine learning basics - use decision trees to make predictions about coupons

In order to get close to real life and applications, the processing of actual data sets is the main focus. From January 1, 2016 to June 30, 2016, real online and offline consumption behaviors are predicted to be used by users within 15 days after receiving coupons in July 2016.
Note: In order to protect the privacy of users and businesses, all data is anonymized, and biased sampling and necessary filtering are used.



**Data set ccf_offline_stage1_train.csv (training data)**

Field | Description
:-|-
User_id | 用户ID
Merchant_id | 商户ID
Coupon_id | 优惠券ID：null表示无优惠券消费，此时Discount_rate和Date_received字段无意义
Discount_rate | 优惠率：x \in [0,1]代表折扣率；x:y表示满x减y。单位是元
Distance | user经常活动的地点离该merchant的最近门店距离是x*500米（如果是连锁店，则取最近的一家门店），x$\in[0,10]$；null表示无此信息，0表示低于500米，10表示大于5公里；
Date_received | 领取优惠券日期
Date | 消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；

```python
# load plugin
import pandas as pd
import numpy as np

#  load data
train_data = pd.read_csv('~/data/ccf_offline_stage1_train.csv')
train_data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1754884 entries, 0 to 1754883
Data columns (total 7 columns):
 #   Column         Dtype  
---  ------         -----  
 0   User_id        int64  
 1   Merchant_id    int64  
 2   Coupon_id      float64
 3   Discount_rate  object 
 4   Distance       float64
 5   Date_received  float64
 6   Date           float64
dtypes: float64(4), int64(2), object(1)
memory usage: 93.7+ MB
"""

train_data.head()
"""
	User_id	Merchant_id	Coupon_id	Discount_rate	Distance	Date_received	Date
0	1439408	2632	NaN	NaN	0.0	NaN	20160217.0
1	1439408	4663	11002.0	150:20	1.0	20160528.0	NaN
2	1439408	2632	8591.0	20:1	0.0	20160217.0	NaN
3	1439408	2632	1078.0	20:1	0.0	20160319.0	NaN
4	1439408	2632	8591.0	20:1	0.0	20160613.0	NaN
"""

print(train_data.shape)

data = train_data.dropna(how = 'any')
print(train_data.shape)
"""
(1754884, 7)
(1754884, 7)
"""
```



`Discount_rate`是object类型的，object在pandas中代表字符串，字符串类型不能输入模型中，所以需要改为数值类型

```python
print('Discount_rate 类型: \n', data['Discount_rate'].unique())
# [0,1] 表示折扣率
# x:y 表示满 x 减 y

"""
Discount_rate 类型: 
 ['20:1' '20:5' '30:5' '50:10' '10:5' '50:20' '100:10' '30:10' '50:5'
 '30:1' '100:30' '0.8' '200:30' '100:20' '10:1' '200:20' '0.95' '5:1'
 '100:5' '100:50' '50:1' '20:10' '150:10' '0.9' '200:50' '150:20' '150:50'
 '200:5' '300:30' '100:1' '200:10' '150:30' '0.85' '0.6' '0.5' '300:20'
 '200:100' '300:50' '150:5' '300:10' '0.75' '0.7' '30:20' '50:30']
"""
```



**Convert Discount_rate into numerical features**

Discount type

x:y 表示满 x 减 y          将 x:y 类型的字符串设为1

 [0,1] 表示折扣率           将 [0,1] 类型的字符串设为 0 

```python
def getDiscountType(row):
    if ':' in row:
        return 1
    else:
        return 0

data['Discount_rate'] = data['Discount_rate'].apply(getDiscountType)
"""
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  import sys
"""

data.info()
"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 67165 entries, 6 to 1754880
Data columns (total 7 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   User_id        67165 non-null  int64  
 1   Merchant_id    67165 non-null  int64  
 2   Coupon_id      67165 non-null  float64
 3   Discount_rate  67165 non-null  int64  
 4   Distance       67165 non-null  float64
 5   Date_received  67165 non-null  float64
 6   Date           67165 non-null  float64
dtypes: float64(4), int64(3)
memory usage: 4.1 MB
"""

# load plugin
# Import DecisionTreeClassifier model
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split, used to divide the data set and test set
from sklearn.model_selection import train_test_split
# Import accuracy_score accuracy index
from sklearn.metrics import accuracy_score
```



add label row to the dataset

Labeling Label Label which samples are positive samples y=1 and which are negative samples y = -1
Forecast goal: the user's consumption within 15 days after receiving the coupon
(Date-Date_received <= 15) means to receive the coupon and use it within 15 days, that is, a positive sample, y = 1
(Date-Date_received> 15) means that the coupon has not been used within 15 days, that is, a negative sample, y = 0
pandas tutorial on time ```https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html```



```python
def label(row):
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format = '%Y%m%d') - pd.to_datetime(row['Date_received'], format = '%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

data['label'] = data.apply(label, axis = 1)
"""
See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
"""
```



Statistics positive and negative samples

```python
print(data['label'].value_counts())
"""
1    57060
0    10105
Name: label, dtype: int64
"""
```

Divide the data set 80% training set 20% test set

80% train 20% test

```python
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size = 0.2, random_state=3)
```

Check the number and category distribution of training samples

```python
y_train.value_counts()
"""
2751537    96
6641735    86
           ..
4461556     1
Name: User_id, Length: 34984, dtype: int64
"""
```

Check the number and type distribution of test samples

```python
y_test.value_counts()
"""
6641735    27
2751537    22
           ..
89464       1
Name: User_id, Length: 11405, dtype: int64
"""
```

Initialize the classification decision tree model, the depth is 5 layers

```python
model = DecisionTreeClassifier(max_depth=6, random_state = 1)
```

Model training

```python
model.fit(X_train, y_train)
```

Model prediction

```python
y_pred = model.predict(X_test)
```

Model evaluation

```python
accuracy_score(y_test, y_pred)
"""
0.011315417256011316
"""
```

Change the standard of the model selection feature to entropy

```python
model = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=2)
```

Model training

```python
model.fit(X_train, y_train)
```

predict

```python
y_pred = model.predict(X_test)
```

Evaluate

```python
accuracy_score(y_test, y_pred)
```



> In addition to the above key steps, you can explore the data by yourself, as well as any other forms of feature preprocessing methods and feature engineering processing. I hope to focus on understanding the development process of machine learning tasks. For the skills and methods of data processing, it is encouraged to invest more time to explore.


## iris

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz 

iris = load_iris()
X = iris.data 
y = iris.target
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X, y)

export_graphviz(
            tree_clf,
            out_file="~/data/course_data/iris_tree.dot",
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            rounded=True,
            filled=True)

for line in open('~/data/course_data/iris_tree.dot'):
    print(line)
"""
digraph Tree {

node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;

edge [fontname=helvetica] ;

0 [label="petal length (cm) <= 2.45\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]\nclass = setosa", fillcolor="#ffffff"] ;

1 [label="gini = 0.0\nsamples = 50\nvalue = [50, 0, 0]\nclass = setosa", fillcolor="#e58139"] ;

0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;

2 [label="petal width (cm) <= 1.75\ngini = 0.5\nsamples = 100\nvalue = [0, 50, 50]\nclass = versicolor", fillcolor="#ffffff"] ;

0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;

3 [label="petal length (cm) <= 4.95\ngini = 0.168\nsamples = 54\nvalue = [0, 49, 5]\nclass = versicolor", fillcolor="#4de88e"] ;

2 -> 3 ;

4 [label="petal width (cm) <= 1.65\ngini = 0.041\nsamples = 48\nvalue = [0, 47, 1]\nclass = versicolor", fillcolor="#3de684"] ;

3 -> 4 ;

5 [label="gini = 0.0\nsamples = 47\nvalue = [0, 47, 0]\nclass = versicolor", fillcolor="#39e581"] ;
show more (open the raw output data in a text editor) ...

16 [label="gini = 0.0\nsamples = 43\nvalue = [0, 0, 43]\nclass = virginica", fillcolor="#8139e5"] ;

12 -> 16 ;

}
"""
```

### Salient Features


```python
tree_clf.feature_importances_
```
### Build Decision Tree: CART
```python
import pandas as pd

mock_data = {
    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}
dataset = pd.DataFrame.from_dict(mock_data)


import numpy as np
from collections import Counter

def entropy(elements):
    counter = Counter(elements)
    probabilities = [counter[e] / len(elements) for e in elements]
    return -sum(p * np.log10(p) for p in probabilities)

def find_the_min_spilter(training_data: pd.DataFrame, target: str) -> str:
    x_fields = set(training_data.columns.tolist()) - {target}
    
    spliter = None
    min_entropy = float('inf')
    
    for f in x_fields:
        elements = set(training_data[f])
        for e in elements:
            sub_spliter_1 = training_data[dataset[f] == e][target].tolist()
            entropy_1 = entropy(sub_spliter_1)
            sub_spliter_2 = training_data[dataset[f] != e][target].tolist()
            entropy_2 = entropy(sub_spliter_2)
            entropy_v = entropy_1 + entropy_2
            
            if entropy_v < min_entropy:
                min_entropy = entropy_v
                spliter = (f, e)
    
    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    
    return spliter

find_the_min_spilter(dataset, 'bought')
"""
spliter is: ('income', '+10')
the min entropy is: 0.7176797562470717
('income', '+10')
"""

dataset[dataset['income'] == '-10']
"""
	gender	income	family_number	bought
1	    F	    -10	            1	    1
6	    M	    -10         	2	    1
"""

dataset[dataset['income'] != '-10']
"""
	gender	income	family_number	bought
0	    F	    +10         	1   	1
2	    F	    +10         	2   	1
3	    F	    +10         	1   	0
4	    M	    +10         	1   	0
5	    M	    +10         	1   	0
"""
```









