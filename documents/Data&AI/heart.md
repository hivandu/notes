# Logistic regression to diagnose heart disease

> The preject source code url : [Heart](https://github.com/hivandu/colab/blob/master/AI_Data/heart.ipynb)

## load data

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/heart.csv')
# the csv url: https://github.com/hivandu/colab/blob/master/AI_Data/data/heart.csv

# Print a brief summary of the data set

data.info()
data.shape

data.target.value_counts()
```

## The params meaning

```
Params	  Meaning	
age	      年龄	
sex	      性别(1 = 男性, 0 = 女性)
cp	      胸部疼痛类型(值1：典型心绞痛，值2：非典型性心绞痛，值3：非心绞痛，值4：无症状）
trestbps   血压	
chol	   胆固醇	
fbs	      空腹血糖（> 120 mg/dl，1=真；0=假）	
restecg	   心电图结果（0=正常，1=患有ST-T波异常，2=根据Estes的标准显示可能或确定的左心室肥大）	
thalach	   最大心跳数	
exang	   运动时是否心绞痛（1=有过；0=没有）
oldpeak	   运动相对于休息的ST
slop	   心电图ST segment的倾斜度(值1:上坡，值2:平坦，值3:下坡） 	
ca	      透视检查看到的血管数	
thal	   缺陷种类（3=正常；6=固定缺陷；7=可逆缺陷）
target	   是否患病（0=否，1=是）
```

## Perform analysis

```python

# Change the "sex" column into two columns "sex_0" and "sex_1"
sex = pd.get_dummies(data['sex'], prefix = 'sex')        

# Add "sex_0" and "sex_1" to the data set. 
data = pd.concat([data, sex], axis = 1)


# And delete the sex column.
data = data.drop(columns = ['sex'])


# Print out the first five lines. Check whether sex_0, sex_1 are added successfully, and whether sex is deleted successfully.
data.head()

# Get sample label
data_y = data.target.values
data_y.shape

# Get sample feature set
data_x = data.drop(['target'], axis = 1)
data_x.shape

# Divide the data set
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.3, random_state=33)
```

### Normalization

```python
# initialize
ss = StandardScaler()

# The fit function/module is used to train model parameters
ss.fit(train_x)

# Standardize the training set and test set
train_x = ss.transform(train_x)
test_x = ss.transform(test_x)

# Define a logistic regression model
lr = LogisticRegression()
lr.fit(train_x, train_y)

# Calculate the training set score
lr.score(train_x, train_y)

# Calculate test set score
lr.score(test_x, test_y)

# Use the classification_report function to display a text report of the main classification indicators
predict = lr.predict(test_x)
print(classification_report(test_y, predict))
```